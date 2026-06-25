from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


# Length-bucket boundaries. These match the bins used in
# `results/ast_kv_distance/` and `results/same_code_context_variation/`.
_LENGTH_BIN_EDGES = (50, 200, 500)

# Coarser bins used by the context_aware_confidence modifier. The position
# offset is bucketed because we only have 6 sample offsets (0,5,10,25,50,100)
# and need a stable lookup for the runtime gate.
_POSITION_OFFSET_BINS = ("0", "5-25", "50-100")

# The 4 system prompt classes the experiment was run with; matches the keys
# in `MAScoder/src/mascoder/prompts.py` (planner / implementer / reviewer /
# tester). Unknown classes fall back to "planner" at lookup time.
_KNOWN_SYSTEM_PROMPT_CLASSES = ("planner", "coder", "reviewer", "tester")

# The 4 surrounding_code_class values produced by the context_sampler. These
# strings are part of the predicted_distance_table.json key, so they must
# match exactly.
_KNOWN_SURROUNDING_CLASSES = ("none", "class_wrap", "try_wrap", "imports_wrap")


def length_bin_for(token_count: int) -> str:
    if token_count < _LENGTH_BIN_EDGES[0]:
        return "<50"
    if token_count < _LENGTH_BIN_EDGES[1]:
        return "50-200"
    if token_count < _LENGTH_BIN_EDGES[2]:
        return "200-500"
    return ">500"


def _position_offset_bin(offset: int) -> str:
    """Bucket raw token offset into one of the 3 bins used in the table."""
    if offset <= 0:
        return "0"
    if offset <= 25:
        return "5-25"
    return "50-100"


def context_aware_confidence_enabled() -> bool:
    """Whether the data-driven context_aware_confidence modifier is active.

    Opt-in via SGLANG_CONTEXT_AWARE_CONFIDENCE=1. Default ON once the
    `predicted_distance_table.json` artifact exists at the default path.
    """
    if "SGLANG_CONTEXT_AWARE_CONFIDENCE" in os.environ:
        return os.environ["SGLANG_CONTEXT_AWARE_CONFIDENCE"] == "1"
    # Auto-enable if the table file is present.
    return _default_table_path().exists()


def _default_table_path() -> Path:
    return Path(__file__).resolve().parents[4] / "results" / "same_code_context_variation" / "data" / "predicted_distance_table.json"


@lru_cache(maxsize=1)
def _load_context_distance_table() -> dict:
    """Load the predicted_distance_table.json produced by
    `results/same_code_context_variation/distance_table_builder.py`.

    If the file is missing (e.g. the experiment hasn't been run), returns a
    safe no-op table that maps everything to the baseline confidence
    multiplier of 1.0. This keeps the gate logic working in development and
    unit tests even when the data isn't yet available.
    """
    path_str = os.environ.get("SGLANG_CONTEXT_DISTANCE_TABLE")
    if path_str:
        path = Path(path_str)
    else:
        path = _default_table_path()
    if not path.exists():
        return {
            "schema_version": "v0-missing",
            "cells": [],
            "global": {
                "predicted_d_norm_baseline": 1.0,
                "predicted_d_norm_max_observed": 1.0,
            },
        }
    with open(path) as f:
        return json.load(f)


def reset_context_distance_table_cache() -> None:
    """Clear the cached table (used by tests and after re-running the
    experiment)."""
    _load_context_distance_table.cache_clear()


@dataclass
class AnchorMetadata:
    code_anchor_signature: str = ""
    code_content_signature: str = ""
    code_anchor_spans: list[dict[str, Any]] = field(default_factory=list)
    reuse_mode: str = ""
    lossy_alignment_method: str = ""
    template_task_family: str = ""
    template_workflow_signature: str = ""
    template_structural_fingerprint: str = ""
    token_count: int = 0           # approximate; used to derive length_bin
    length_bin: str = ""           # <50 | 50-200 | 200-500 | >500
    # Prompt-context fields (results/same_code_context_variation). These
    # describe WHERE the code is sent (not what the code is) and are used by
    # the context_aware_confidence modifier to predict KV reuse quality for
    # an exact-content match.
    nesting_depth: int = 0
    prompt_position_offset: int = 0
    system_prompt_class: str = ""
    surrounding_code_hash: str = ""


@dataclass
class AnchorMatchResult:
    reuse_allowed: bool
    reuse_confidence: float
    matched_anchor_signature: str = ""
    matched_content_signature: str = ""
    syntax_region_type: str = ""
    match_reason: str = ""
    rejected_reason: str | None = None
    # Context-aware modifier telemetry (sglang-kvflow context_aware_confidence)
    predicted_distance: float = 0.0         # the d_norm predicted for this (code, context) bucket
    context_aware_multiplier: float = 1.0   # multiplier applied to base confidence
    structural_distance_hint: float = 0.0   # legacy field, kept for telemetry backwards-compat


def normalize_anchor_spans(spans: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in spans or []:
        if not isinstance(raw, dict):
            continue
        anchor_type = str(raw.get("anchor_type", "") or "").strip()
        signature = str(raw.get("signature", "") or "").strip()
        if not anchor_type and not signature:
            continue
        try:
            start_line = int(raw.get("start_line", 0) or 0)
        except (TypeError, ValueError):
            start_line = 0
        try:
            end_line = int(raw.get("end_line", 0) or 0)
        except (TypeError, ValueError):
            end_line = 0
        if start_line and end_line and end_line < start_line:
            start_line, end_line = end_line, start_line
        normalized.append(
            {
                "anchor_type": anchor_type,
                "signature": signature,
                "content_signature": str(raw.get("content_signature", "") or "").strip(),
                "start_line": start_line,
                "end_line": end_line,
            }
        )
    return normalized


def build_anchor_metadata(
    *,
    code_anchor_signature: str = "",
    code_content_signature: str = "",
    code_anchor_spans: Iterable[dict[str, Any]] | None = None,
    reuse_mode: str = "",
    lossy_alignment_method: str = "",
    template_task_family: str = "",
    template_workflow_signature: str = "",
    template_structural_fingerprint: str = "",
    token_count: int = 0,
    length_bin: str = "",
    # Prompt-context fields (sglang-kvflow context_aware_confidence)
    nesting_depth: int = 0,
    prompt_position_offset: int = 0,
    system_prompt_class: str = "",
    surrounding_code_hash: str = "",
) -> AnchorMetadata:
    if not length_bin and token_count:
        length_bin = length_bin_for(token_count)
    return AnchorMetadata(
        code_anchor_signature=str(code_anchor_signature or ""),
        code_content_signature=str(code_content_signature or ""),
        code_anchor_spans=normalize_anchor_spans(code_anchor_spans),
        reuse_mode=str(reuse_mode or ""),
        lossy_alignment_method=str(lossy_alignment_method or ""),
        template_task_family=str(template_task_family or ""),
        template_workflow_signature=str(template_workflow_signature or ""),
        template_structural_fingerprint=str(template_structural_fingerprint or ""),
        token_count=int(token_count or 0),
        length_bin=str(length_bin or ""),
        nesting_depth=int(nesting_depth or 0),
        prompt_position_offset=int(prompt_position_offset or 0),
        system_prompt_class=str(system_prompt_class or ""),
        surrounding_code_hash=str(surrounding_code_hash or ""),
    )


def _content_signatures(meta: AnchorMetadata) -> set[str]:
    signatures = {meta.code_content_signature} if meta.code_content_signature else set()
    for span in meta.code_anchor_spans:
        content_signature = str(span.get("content_signature", "") or "")
        if content_signature:
            signatures.add(content_signature)
    return signatures


def match_request_to_candidate(request: AnchorMetadata, candidate: AnchorMetadata) -> AnchorMatchResult:
    if request.reuse_mode != "lossy":
        return AnchorMatchResult(
            reuse_allowed=False,
            reuse_confidence=0.0,
            rejected_reason="reuse_mode_disabled",
        )
    if not request.code_anchor_signature and not request.code_anchor_spans:
        return AnchorMatchResult(
            reuse_allowed=False,
            reuse_confidence=0.0,
            rejected_reason="missing_request_anchor",
        )
    if not candidate.code_anchor_signature and not candidate.code_anchor_spans:
        return AnchorMatchResult(
            reuse_allowed=False,
            reuse_confidence=0.0,
            rejected_reason="missing_candidate_anchor",
        )
    request_content_signatures = _content_signatures(request)
    candidate_content_signatures = _content_signatures(candidate)
    shared_content_signatures = request_content_signatures & candidate_content_signatures
    if not request_content_signatures:
        return AnchorMatchResult(
            reuse_allowed=False,
            reuse_confidence=0.0,
            rejected_reason="missing_request_content_signature",
        )
    if not candidate_content_signatures:
        return AnchorMatchResult(
            reuse_allowed=False,
            reuse_confidence=0.0,
            rejected_reason="missing_candidate_content_signature",
        )
    matched_content_signature = sorted(shared_content_signatures)[0] if shared_content_signatures else ""

    if not shared_content_signatures:
        return AnchorMatchResult(
            reuse_allowed=False,
            reuse_confidence=0.0,
            rejected_reason="code_content_signature_mismatch",
        )
    if (
        request.lossy_alignment_method
        and candidate.lossy_alignment_method
        and request.lossy_alignment_method != candidate.lossy_alignment_method
    ):
        return AnchorMatchResult(
            reuse_allowed=False,
            reuse_confidence=0.0,
            rejected_reason="alignment_method_mismatch",
        )
    if (
        request.template_task_family
        and candidate.template_task_family
        and request.template_task_family != candidate.template_task_family
    ):
        return AnchorMatchResult(
            reuse_allowed=False,
            reuse_confidence=0.0,
            rejected_reason="template_task_family_mismatch",
        )

    structural_conflict = (
        request.template_structural_fingerprint
        and candidate.template_structural_fingerprint
        and request.template_structural_fingerprint != candidate.template_structural_fingerprint
    )
    workflow_conflict = (
        request.template_workflow_signature
        and candidate.template_workflow_signature
        and request.template_workflow_signature != candidate.template_workflow_signature
    )

    if shared_content_signatures:
        # context_aware_confidence modifier: keep the 0.95 base confidence
        # for an exact-content match, but multiply it down based on the
        # predicted KV distance for the REQUEST's prompt context. When the
        # prediction is high (large position offset, very different system
        # prompt, etc.), the modifier can drop the match below 0.5 to refuse
        # the reuse. See results/same_code_context_variation/ for the data.
        base_confidence = 0.95
        if context_aware_confidence_enabled():
            predicted_d, multiplier, allowed, demoted = _apply_context_aware_confidence(
                request, base_confidence=base_confidence,
            )
        else:
            predicted_d, multiplier, demoted = 0.0, 1.0, False
        new_confidence = round(base_confidence * multiplier, 4)
        if not context_aware_confidence_enabled():
            # Modifier disabled: keep the original behaviour unchanged.
            new_confidence = base_confidence
            allowed = True
        return AnchorMatchResult(
            reuse_allowed=allowed,
            reuse_confidence=new_confidence,
            matched_anchor_signature=candidate.code_anchor_signature,
            matched_content_signature=matched_content_signature,
            syntax_region_type=_preferred_region_type(request.code_anchor_spans, candidate.code_anchor_spans),
            match_reason="exact_code_content_signature" if not demoted else "exact_code_content_signature_demoted",
            rejected_reason=None if allowed else "context_aware_confidence_below_floor",
            predicted_distance=predicted_d,
            context_aware_multiplier=round(multiplier, 4),
        )

    if (
        request.code_anchor_signature
        and candidate.code_anchor_signature
        and request.code_anchor_signature == candidate.code_anchor_signature
    ):
        confidence = 1.0
        if workflow_conflict:
            confidence *= 0.9
        if structural_conflict:
            confidence *= 0.85
        return AnchorMatchResult(
            reuse_allowed=True,
            reuse_confidence=round(confidence, 4),
            matched_anchor_signature=candidate.code_anchor_signature,
            matched_content_signature=matched_content_signature,
            syntax_region_type=_preferred_region_type(request.code_anchor_spans, candidate.code_anchor_spans),
            match_reason="exact_anchor_signature",
        )

    best_overlap = 0.0
    best_type = ""
    best_signature = candidate.code_anchor_signature
    for request_span in request.code_anchor_spans:
        for candidate_span in candidate.code_anchor_spans:
            if request_span.get("anchor_type") != candidate_span.get("anchor_type"):
                continue
            req_content = str(request_span.get("content_signature", "") or "")
            cand_content = str(candidate_span.get("content_signature", "") or "")
            if req_content and cand_content and req_content != cand_content:
                continue
            overlap = _span_similarity(request_span, candidate_span)
            if overlap > best_overlap:
                best_overlap = overlap
                best_type = str(request_span.get("anchor_type", "") or "")
                best_signature = str(candidate_span.get("signature", "") or best_signature)

    if best_overlap >= 0.8:
        confidence = 0.82
        reason = "span_overlap_high"
    elif best_overlap >= 0.5:
        confidence = 0.68
        reason = "span_overlap_medium"
    elif best_overlap >= 0.3:
        confidence = 0.55
        reason = "span_overlap_low"
    else:
        return AnchorMatchResult(
            reuse_allowed=False,
            reuse_confidence=0.0,
            rejected_reason="no_anchor_overlap",
        )

    if workflow_conflict:
        confidence *= 0.85
        reason = f"{reason}+workflow_mismatch_penalty"

    return AnchorMatchResult(
        reuse_allowed=confidence >= 0.5,
        reuse_confidence=round(confidence, 4),
        matched_anchor_signature=best_signature,
        matched_content_signature=matched_content_signature,
        syntax_region_type=best_type or _preferred_region_type(request.code_anchor_spans, candidate.code_anchor_spans),
        match_reason=reason,
        rejected_reason=None if confidence >= 0.5 else "low_reuse_confidence",
    )


def select_best_match(
    request: AnchorMetadata,
    candidates: Iterable[tuple[object, AnchorMetadata]],
) -> tuple[object | None, AnchorMatchResult]:
    best_candidate: object | None = None
    best_result = AnchorMatchResult(
        reuse_allowed=False,
        reuse_confidence=0.0,
        rejected_reason="no_candidates",
    )
    for candidate_ref, candidate_meta in candidates:
        result = match_request_to_candidate(request, candidate_meta)
        if result.reuse_allowed:
            if result.reuse_confidence > best_result.reuse_confidence:
                best_candidate = candidate_ref
                best_result = result
            elif (
                result.reuse_confidence == best_result.reuse_confidence
                and best_result.reuse_allowed
                and best_candidate is not None
            ):
                # Prefer deeper nodes (more KV tokens) for fuzzy matching potential
                cand_len = (
                    len(candidate_ref.key)
                    if hasattr(candidate_ref, "key") and candidate_ref.key
                    else 0
                )
                best_len = (
                    len(best_candidate.key)
                    if hasattr(best_candidate, "key") and best_candidate.key
                    else 0
                )
                if cand_len > best_len:
                    best_candidate = candidate_ref
                    best_result = result
        elif (
            best_candidate is None
            and not best_result.reuse_allowed
            and best_result.rejected_reason == "no_candidates"
        ):
            best_result = result
    return best_candidate, best_result


def _lookup_predicted_d_norm(
    table: dict,
    length_bin: str,
    position_offset_bin: str,
    system_prompt_class: str,
    surrounding_code_class: str,
) -> float | None:
    """Return the predicted d_norm for a (length, offset, system, surround)
    cell, or None if the cell is missing. Buckets unknown values back to
    the planner/none defaults rather than rejecting — the modifier is
    advisory, not gating.
    """
    sys_cls = system_prompt_class if system_prompt_class in _KNOWN_SYSTEM_PROMPT_CLASSES else "planner"
    surr_cls = surrounding_code_class if surrounding_code_class in _KNOWN_SURROUNDING_CLASSES else "none"
    lb = length_bin if length_bin in ("<50", "50-200", "200-500", ">500") else "50-200"
    for cell in table.get("cells", []):
        if (
            cell.get("length_bin") == lb
            and cell.get("position_offset") == position_offset_bin
            and cell.get("system_prompt_class") == sys_cls
            and cell.get("surrounding_code_class") == surr_cls
        ):
            return cell.get("predicted_d_norm_mean")
    return None


def _apply_context_aware_confidence(
    request: AnchorMetadata,
    *,
    base_confidence: float = 0.95,
) -> tuple[float, float, bool, bool]:
    """Compute the data-driven confidence modifier for an exact-content match.

    Returns (predicted_d, multiplier, allowed, demoted):
        predicted_d  — the d_norm we predict for this (code, context) bucket
        multiplier   — the value to multiply base_confidence by
        allowed      — whether the modified confidence still meets the 0.5 floor
        demoted      — True if the modifier dropped us below 0.5 (caller should
                       mark match_reason as `_demoted`)

    The modifier is opt-in (env SGLANG_CONTEXT_AWARE_CONFIDENCE). When the
    predicted_distance_table.json is missing, the no-op table in
    _load_context_distance_table returns multiplier=1.0 and predicted_d=0.0,
    so the function safely no-ops.
    """
    table = _load_context_distance_table()
    pos_bin = _position_offset_bin(request.prompt_position_offset)
    sys_cls = request.system_prompt_class or "planner"
    surr_cls = "none"   # surrounding_code_hash is hashed; we don't have class info on this side
    # We don't bucket on surrounding_code_class from the request side (it's
    # stored as a hash). When the surrounding_code_hash is missing, fall back
    # to the none class. When present, keep it as "none" unless the client
    # also sent a class — we don't decode the hash.
    predicted_d = _lookup_predicted_d_norm(
        table, request.length_bin or "50-200", pos_bin, sys_cls, surr_cls,
    )
    if predicted_d is None:
        predicted_d = table.get("global", {}).get("predicted_d_norm_baseline", 1.0)
    max_allowed_predicted = os.environ.get("SGLANG_CONTEXT_AWARE_MAX_PREDICTED_D")
    if max_allowed_predicted:
        try:
            threshold = float(max_allowed_predicted)
        except ValueError:
            threshold = 0.0
        if threshold > 0 and predicted_d > threshold:
            return predicted_d, 0.0, False, True
    d_max = max(table.get("global", {}).get("predicted_d_norm_max_observed", 1.0), 1e-6)
    multiplier = 0.5 + 0.5 * max(0.0, 1.0 - predicted_d / d_max)
    new_conf = base_confidence * multiplier
    demoted = new_conf < 0.5
    return predicted_d, multiplier, not demoted, demoted


def _preferred_region_type(
    request_spans: Iterable[dict[str, Any]],
    candidate_spans: Iterable[dict[str, Any]],
) -> str:
    for span in list(request_spans) + list(candidate_spans):
        anchor_type = str(span.get("anchor_type", "") or "")
        if anchor_type:
            return anchor_type
    return "code_anchor"


MAX_EFFECTIVE_SPAN = 5


def _span_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start = int(left.get("start_line", 0) or 0)
    left_end = int(left.get("end_line", 0) or 0)
    right_start = int(right.get("start_line", 0) or 0)
    right_end = int(right.get("end_line", 0) or 0)
    if not left_start or not left_end or not right_start or not right_end:
        return 0.0
    left_len = left_end - left_start + 1
    right_len = right_end - right_start + 1
    # Cap effective span: large code blocks should not dominate IoU
    left_eff = min(left_len, MAX_EFFECTIVE_SPAN)
    right_eff = min(right_len, MAX_EFFECTIVE_SPAN)
    # Compute intersection with capped spans (both starting from 0)
    intersection = min(left_eff, right_eff)
    if intersection <= 0:
        return 0.0
    base_iou = intersection / max(left_eff, right_eff)
    # If both blocks are large (>cap), reduce similarity: span size alone is not signal
    if left_len > MAX_EFFECTIVE_SPAN and right_len > MAX_EFFECTIVE_SPAN:
        base_iou *= 0.25
    return base_iou


__all__ = [
    "AnchorMatchResult",
    "AnchorMetadata",
    "build_anchor_metadata",
    "context_aware_confidence_enabled",
    "length_bin_for",
    "match_request_to_candidate",
    "normalize_anchor_spans",
    "reset_context_distance_table_cache",
    "select_best_match",
]
