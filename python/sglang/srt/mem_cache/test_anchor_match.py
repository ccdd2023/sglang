from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PYTHON_ROOT = PROJECT_ROOT / "sglang-kvflow" / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.srt.mem_cache.anchor_match import build_anchor_metadata, match_request_to_candidate


def test_match_request_to_candidate_accepts_exact_signature(monkeypatch):
    """The original anchor-match exact-content test. Disables the modifier so
    the base 0.95 confidence is preserved and the existing assertion
    (>= 0.85) holds. With the modifier enabled, the test would also pass
    for canonical (offset=0) requests — but we want to verify both the
    legacy exact-content behaviour AND the modifier's no-op for this case."""
    monkeypatch.setenv("SGLANG_CONTEXT_AWARE_CONFIDENCE", "0")
    monkeypatch.setenv("SGLANG_CONTEXT_DISTANCE_TABLE", "/nonexistent/path.json")
    am = _fresh_anchor_module()
    request = am.build_anchor_metadata(
        code_anchor_signature="sig-1",
        code_content_signature="content-1",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-1", "start_line": 1, "end_line": 6}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
        template_task_family="code_generation",
    )
    candidate = am.build_anchor_metadata(
        code_anchor_signature="sig-1",
        code_content_signature="content-1",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-1", "start_line": 1, "end_line": 6}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
        template_task_family="code_generation",
    )
    result = am.match_request_to_candidate(request, candidate)
    assert result.reuse_allowed is True
    assert result.match_reason == "exact_code_content_signature"
    assert result.matched_content_signature == "content-1"
    assert result.reuse_confidence >= 0.85


def test_match_request_to_candidate_rejects_task_family_mismatch():
    request = build_anchor_metadata(
        code_anchor_signature="sig-2",
        code_content_signature="content-2",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-2", "start_line": 1, "end_line": 6}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
        template_task_family="bug_fix",
    )
    candidate = build_anchor_metadata(
        code_anchor_signature="sig-2",
        code_content_signature="content-2",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-2", "start_line": 1, "end_line": 6}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
        template_task_family="testing",
    )
    result = match_request_to_candidate(request, candidate)
    assert result.reuse_allowed is False
    assert result.rejected_reason == "template_task_family_mismatch"


def test_match_request_to_candidate_accepts_span_overlap_only_with_same_content():
    request = build_anchor_metadata(
        code_anchor_signature="sig-request",
        code_content_signature="content-3",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-request", "start_line": 10, "end_line": 18}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    candidate = build_anchor_metadata(
        code_anchor_signature="sig-candidate",
        code_content_signature="content-3",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-candidate", "start_line": 11, "end_line": 19}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    result = match_request_to_candidate(request, candidate)
    assert result.reuse_allowed is True
    assert result.match_reason == "exact_code_content_signature"
    assert result.matched_content_signature == "content-3"
    assert result.syntax_region_type == "function"


def test_match_request_to_candidate_accepts_segment_content_signature():
    request = build_anchor_metadata(
        code_anchor_signature="request",
        code_anchor_spans=[
            {"anchor_type": "code_base", "signature": "cb1", "content_signature": "content-shared", "start_line": 1, "end_line": 20}
        ],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    candidate = build_anchor_metadata(
        code_anchor_signature="candidate",
        code_anchor_spans=[
            {"anchor_type": "code_base", "signature": "cb1-old", "content_signature": "content-shared", "start_line": 50, "end_line": 69}
        ],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    result = match_request_to_candidate(request, candidate)
    assert result.reuse_allowed is True
    assert result.matched_content_signature == "content-shared"
    assert result.match_reason == "exact_code_content_signature"


def test_match_request_to_candidate_rejects_same_ast_marker_with_different_content():
    request = build_anchor_metadata(
        code_anchor_signature="sig-3",
        code_content_signature="content-a",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-3", "start_line": 1, "end_line": 4}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    candidate = build_anchor_metadata(
        code_anchor_signature="sig-3",
        code_content_signature="content-b",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-3", "start_line": 10, "end_line": 13}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    result = match_request_to_candidate(request, candidate)
    assert result.reuse_allowed is False
    assert result.rejected_reason == "code_content_signature_mismatch"


def test_match_request_to_candidate_rejects_missing_content_signature():
    request = build_anchor_metadata(
        code_anchor_signature="sig-4",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-4", "start_line": 1, "end_line": 4}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    candidate = build_anchor_metadata(
        code_anchor_signature="sig-4",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-4", "start_line": 1, "end_line": 4}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    result = match_request_to_candidate(request, candidate)
    assert result.reuse_allowed is False
    assert result.rejected_reason == "missing_request_content_signature"


def test_match_request_to_candidate_rejects_no_overlap():
    request = build_anchor_metadata(
        code_anchor_signature="sig-request",
        code_content_signature="content-req",
        code_anchor_spans=[{"anchor_type": "function", "signature": "sig-request", "start_line": 1, "end_line": 4}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    candidate = build_anchor_metadata(
        code_anchor_signature="sig-candidate",
        code_content_signature="content-cand",
        code_anchor_spans=[{"anchor_type": "class", "signature": "sig-candidate", "start_line": 20, "end_line": 40}],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
    )
    result = match_request_to_candidate(request, candidate)
    assert result.reuse_allowed is False
    # Either structural gate is OFF and we get content-mismatch, or it's ON
    # and we still get rejected because anchor_types differ.
    assert result.rejected_reason in {"code_content_signature_mismatch", "no_anchor_overlap"}


# ---------------------------------------------------------------------------
# context_aware_confidence modifier (data-driven from
# results/same_code_context_variation/)
# ---------------------------------------------------------------------------
import os
import importlib
import json
import tempfile


def _fresh_anchor_module():
    """Reload anchor_match with the current env vars so the modifier toggle
    takes effect mid-test."""
    import sglang.srt.mem_cache.anchor_match as m
    importlib.reload(m)
    return m


def _inject_test_table(am, cells, global_dict=None):
    """Write a tiny predicted_distance_table.json to a temp file and point
    SGLANG_CONTEXT_DISTANCE_TABLE at it. Returns the path so the caller can
    clean up later (or just let pytest tempdir cleanup handle it)."""
    global_dict = global_dict or {
        "predicted_d_norm_baseline": 1.0,
        "predicted_d_norm_max_observed": 2.4,
    }
    table = {
        "schema_version": "v1-test",
        "cells": cells,
        "global": global_dict,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        path = f.name
    os.environ["SGLANG_CONTEXT_DISTANCE_TABLE"] = path
    am.reset_context_distance_table_cache()
    return path


def test_context_aware_confidence_disabled_when_table_missing(monkeypatch):
    """Without a table file, the modifier is a no-op. This keeps the
    production path safe when the experiment hasn't been run yet."""
    monkeypatch.setenv("SGLANG_CONTEXT_AWARE_CONFIDENCE", "0")
    monkeypatch.delenv("SGLANG_CONTEXT_DISTANCE_TABLE", raising=False)
    am = _fresh_anchor_module()
    # Override the default path to point at a non-existent file.
    monkeypatch.setenv("SGLANG_CONTEXT_DISTANCE_TABLE", "/nonexistent/path.json")
    spans = [{"anchor_type": "function", "signature": "s", "content_signature": "c",
              "start_line": 1, "end_line": 8}]
    request = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
        prompt_position_offset=100, system_prompt_class="tester",
    )
    candidate = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
    )
    result = am.match_request_to_candidate(request, candidate)
    assert result.match_reason == "exact_code_content_signature"
    assert result.reuse_confidence == 0.95
    assert result.context_aware_multiplier == 1.0


def test_context_aware_confidence_reduces_when_high_predicted_distance(monkeypatch):
    monkeypatch.setenv("SGLANG_CONTEXT_AWARE_CONFIDENCE", "1")
    am = _fresh_anchor_module()
    # Inject a table with a high predicted_d_norm for offset=50-100 + tester.
    _inject_test_table(am, [
        {"length_bin": "50-200", "position_offset": "50-100",
         "system_prompt_class": "tester", "surrounding_code_class": "none",
         "predicted_d_norm_mean": 2.0, "predicted_d_norm_std": 0.2, "n_samples": 5},
    ])
    spans = [{"anchor_type": "function", "signature": "s", "content_signature": "c",
              "start_line": 1, "end_line": 8}]
    request = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
        prompt_position_offset=100, system_prompt_class="tester", length_bin="50-200",
    )
    candidate = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
    )
    result = am.match_request_to_candidate(request, candidate)
    # multiplier = 0.5 + 0.5 * (1 - 2.0/2.4) = 0.5833
    # confidence = 0.95 * 0.5833 ≈ 0.554
    assert result.reuse_allowed is True
    assert result.context_aware_multiplier < 1.0
    assert result.reuse_confidence < 0.95
    assert result.reuse_confidence > 0.5
    assert result.predicted_distance == 2.0
    assert result.match_reason == "exact_code_content_signature"


def test_context_aware_confidence_demotes_below_floor_for_extreme_predicted(monkeypatch):
    """Synthetic table where d_max = d_predicted → multiplier = 0.5 → confidence = 0.475 < 0.5."""
    monkeypatch.setenv("SGLANG_CONTEXT_AWARE_CONFIDENCE", "1")
    am = _fresh_anchor_module()
    _inject_test_table(am, [
        {"length_bin": "50-200", "position_offset": "50-100",
         "system_prompt_class": "tester", "surrounding_code_class": "none",
         "predicted_d_norm_mean": 1.5, "predicted_d_norm_std": 0.1, "n_samples": 5},
    ], global_dict={"predicted_d_norm_baseline": 1.0, "predicted_d_norm_max_observed": 1.5})
    spans = [{"anchor_type": "function", "signature": "s", "content_signature": "c",
              "start_line": 1, "end_line": 8}]
    request = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
        prompt_position_offset=100, system_prompt_class="tester", length_bin="50-200",
    )
    candidate = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
    )
    result = am.match_request_to_candidate(request, candidate)
    # multiplier = 0.5 + 0.5 * (1 - 1.5/1.5) = 0.5
    # confidence = 0.95 * 0.5 = 0.475 < 0.5
    assert result.reuse_allowed is False
    assert result.rejected_reason == "context_aware_confidence_below_floor"
    assert result.match_reason == "exact_code_content_signature_demoted"
    assert result.context_aware_multiplier == 0.5


def test_context_aware_confidence_does_not_affect_non_content_match(monkeypatch):
    """Modifier only runs after exact_code_content_signature matches. A span-overlap
    tier or content-mismatch rejection should NOT be touched by the modifier."""
    monkeypatch.setenv("SGLANG_CONTEXT_AWARE_CONFIDENCE", "1")
    am = _fresh_anchor_module()
    _inject_test_table(am, [
        {"length_bin": "50-200", "position_offset": "50-100",
         "system_prompt_class": "tester", "surrounding_code_class": "none",
         "predicted_d_norm_mean": 2.4, "predicted_d_norm_std": 0.0, "n_samples": 5},
    ])
    # Different content signatures + different AST types → no content match, no span overlap
    request = am.build_anchor_metadata(
        code_anchor_spans=[{"anchor_type": "function", "signature": "s1", "content_signature": "x",
                            "start_line": 1, "end_line": 8}],
        reuse_mode="lossy", template_task_family="code_gen", token_count=120,
        prompt_position_offset=100, system_prompt_class="tester", length_bin="50-200",
    )
    candidate = am.build_anchor_metadata(
        code_anchor_spans=[{"anchor_type": "class", "signature": "s2", "content_signature": "y",
                            "start_line": 1, "end_line": 8}],
        reuse_mode="lossy", template_task_family="code_gen", token_count=120,
    )
    result = am.match_request_to_candidate(request, candidate)
    # Modifier was not invoked (it only runs on content match)
    assert result.match_reason != "exact_code_content_signature"
    assert result.match_reason != "exact_code_content_signature_demoted"
    assert result.rejected_reason == "code_content_signature_mismatch"


def test_context_aware_confidence_falls_back_to_baseline_when_bucket_missing(monkeypatch):
    """Unknown system_prompt_class or position bin → use the global baseline."""
    monkeypatch.setenv("SGLANG_CONTEXT_AWARE_CONFIDENCE", "1")
    am = _fresh_anchor_module()
    # Table has a cell for offset=0 only. Request asks for offset=100 (50-100 bin).
    _inject_test_table(am, [
        {"length_bin": "50-200", "position_offset": "0",
         "system_prompt_class": "planner", "surrounding_code_class": "none",
         "predicted_d_norm_mean": 1.5, "predicted_d_norm_std": 0.1, "n_samples": 5},
    ], global_dict={"predicted_d_norm_baseline": 1.0, "predicted_d_norm_max_observed": 2.0})
    spans = [{"anchor_type": "function", "signature": "s", "content_signature": "c",
              "start_line": 1, "end_line": 8}]
    request = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
        prompt_position_offset=100,   # "50-100" bin, no cell in table
        system_prompt_class="planner", length_bin="50-200",
    )
    candidate = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
    )
    result = am.match_request_to_candidate(request, candidate)
    # Falls back to baseline = 1.0, multiplier = 0.5 + 0.5 * (1 - 1/2) = 0.75
    assert result.predicted_distance == 1.0
    assert abs(result.context_aware_multiplier - 0.75) < 0.01
    assert result.reuse_allowed is True


def test_context_aware_confidence_field_propagation(monkeypatch):
    """The new fields always appear on the result, even when modifier is no-op."""
    monkeypatch.setenv("SGLANG_CONTEXT_AWARE_CONFIDENCE", "0")
    monkeypatch.setenv("SGLANG_CONTEXT_DISTANCE_TABLE", "/nonexistent/path.json")
    am = _fresh_anchor_module()
    spans = [{"anchor_type": "function", "signature": "s", "content_signature": "c",
              "start_line": 1, "end_line": 8}]
    request = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
    )
    candidate = am.build_anchor_metadata(
        code_anchor_spans=spans, reuse_mode="lossy",
        template_task_family="code_gen", token_count=120,
    )
    result = am.match_request_to_candidate(request, candidate)
    assert result.predicted_distance == 0.0
    assert result.context_aware_multiplier == 1.0


def test_length_bin_for_thresholds():
    am = _fresh_anchor_module()
    assert am.length_bin_for(0) == "<50"
    assert am.length_bin_for(49) == "<50"
    assert am.length_bin_for(50) == "50-200"
    assert am.length_bin_for(199) == "50-200"
    assert am.length_bin_for(200) == "200-500"
    assert am.length_bin_for(499) == "200-500"
    assert am.length_bin_for(500) == ">500"
    assert am.length_bin_for(10_000) == ">500"


# ---------------------------------------------------------------------------
# Regression tests for radix_cache.py bug fixes (2026-06)
# ---------------------------------------------------------------------------
import logging
import threading
from types import SimpleNamespace

import pytest
import torch


def _make_minimal_radix_cache(monkeypatch):
    """Build a RadixCache-like object that has just enough state for the 3
    regression tests below, without spinning up a real model + KV pool.

    We skip __init__ entirely and just attach the 3 attributes that
    _split_node, _decrement_anchor_refs, and _store_anchor_kv touch.
    """
    from sglang.srt.mem_cache import radix_cache as rc

    cache = rc.RadixCache.__new__(rc.RadixCache)
    # Reuse the class's static get_child_key_fn for unit-test simplicity
    cache.get_child_key_fn = rc.get_child_key
    cache.page_size = 1
    cache.anchor_kv_store = {}
    cache.anchor_kv_store_lock = threading.RLock()
    cache.evictable_size_ = 0
    cache.evictable_leaves = set()
    cache.root_node = rc.TreeNode(priority=0)
    return cache


def _make_tree_node(rc, *, anchor_id="", content_sig="", anchor_type="",
                    anchor_spans=None, nesting_depth=0, pos_offset=0,
                    sys_class="", surr_hash=""):
    n = rc.TreeNode(priority=0)
    n.anchor_id = anchor_id
    n.anchor_type = anchor_type
    n.code_content_signature = content_sig
    n.anchor_spans = list(anchor_spans or [])
    n.nesting_depth = nesting_depth
    n.prompt_position_offset = pos_offset
    n.system_prompt_class = sys_class
    n.surrounding_code_hash = surr_hash
    return n


def test_split_node_propagates_anchor_metadata(monkeypatch):
    """Bug A: _split_node must propagate the 8 anchor / context fields from
    the child to the new prefix node, otherwise the prefix becomes
    anchor-blind and select_best_match won't consider it."""
    from sglang.srt.mem_cache import radix_cache as rc
    cache = _make_minimal_radix_cache(monkeypatch)

    tokens = [1, 2, 3, 4, 5, 6]
    child = _make_tree_node(
        rc, anchor_id="aid-1", content_sig="content-1", anchor_type="function",
        anchor_spans=[{"start_line": 1, "end_line": 5}],
        nesting_depth=1, pos_offset=100, sys_class="tester", surr_hash="imports_wrap",
    )
    child.key = rc.RadixKey(tokens)
    child.value = torch.zeros(6, 4, dtype=torch.float32)
    parent = rc.TreeNode(priority=0)
    parent.children = {cache.get_child_key_fn(child.key): child}
    child.parent = parent

    new_node = cache._split_node(child.key, child, split_len=3)
    # The 8 fields must be propagated
    assert new_node.anchor_id == "aid-1"
    assert new_node.anchor_type == "function"
    assert new_node.code_content_signature == "content-1"
    assert new_node.anchor_spans == [{"start_line": 1, "end_line": 5}]
    assert new_node.nesting_depth == 1
    assert new_node.prompt_position_offset == 100
    assert new_node.system_prompt_class == "tester"
    assert new_node.surrounding_code_hash == "imports_wrap"


def test_split_node_does_not_propagate_when_anchor_id_empty(monkeypatch):
    """If the child has no anchor_id, the new_node should also have empty
    anchor fields (not stale copies of garbage)."""
    from sglang.srt.mem_cache import radix_cache as rc
    cache = _make_minimal_radix_cache(monkeypatch)

    tokens = [1, 2, 3, 4, 5, 6]
    child = _make_tree_node(rc)  # all default = empty
    child.key = rc.RadixKey(tokens)
    child.value = torch.zeros(6, 4, dtype=torch.float32)
    parent = rc.TreeNode(priority=0)
    parent.children = {cache.get_child_key_fn(child.key): child}
    child.parent = parent

    new_node = cache._split_node(child.key, child, split_len=3)
    assert new_node.anchor_id == ""
    assert new_node.code_content_signature == ""
    assert new_node.anchor_spans == []


def test_decrement_anchor_refs_drops_entry_at_zero(monkeypatch):
    """Bug B: when a TreeNode carrying an anchor's content_signature is
    deleted, the corresponding AnchorKVEntry's ref_count should drop to 0
    and the entry should be removed from anchor_kv_store."""
    from sglang.srt.mem_cache import radix_cache as rc
    cache = _make_minimal_radix_cache(monkeypatch)

    entry = rc.AnchorKVEntry(
        signature="aid-1", token_ids=[1, 2, 3], kv_indices=None,
        start_pos=0, code_content_signature="content-1",
    )
    entry.ref_count = 1  # baseline
    cache.anchor_kv_store["content-1"] = [entry]

    # Simulate a TreeNode eviction that carries the same content_signature
    node = _make_tree_node(rc, content_sig="content-1")
    cache._decrement_anchor_refs(node)
    assert entry.ref_count == 0
    assert "content-1" not in cache.anchor_kv_store  # dropped


def test_decrement_anchor_refs_keeps_entry_when_refcount_still_positive(monkeypatch):
    """If the entry has been reused elsewhere (ref_count > 1), decrementing
    once should NOT drop it."""
    from sglang.srt.mem_cache import radix_cache as rc
    cache = _make_minimal_radix_cache(monkeypatch)

    entry = rc.AnchorKVEntry(
        signature="aid-1", token_ids=[1, 2, 3], kv_indices=None,
        start_pos=0, code_content_signature="content-1",
    )
    entry.ref_count = 3  # currently used by 2 active borrows
    cache.anchor_kv_store["content-1"] = [entry]

    node = _make_tree_node(rc, content_sig="content-1")
    cache._decrement_anchor_refs(node)
    assert entry.ref_count == 2
    assert cache.anchor_kv_store["content-1"] == [entry]  # still there


def test_store_anchor_kv_warns_on_missing_token_spans(monkeypatch, caplog):
    """Bug C: _store_anchor_kv should log a warning (not silently return)
    when code_anchor_token_spans is missing on a request that has
    code_anchor_signature set."""
    from sglang.srt.mem_cache import radix_cache as rc
    cache = _make_minimal_radix_cache(monkeypatch)

    # Build a minimal req with signature but no token spans
    req = SimpleNamespace(
        rid="test-rid-1",
        code_anchor_signature="aid-1",
        code_content_signature="content-1",
        code_anchor_token_spans=[],   # empty!
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[],
    )

    with caplog.at_level(logging.WARNING, logger="sglang.srt.mem_cache.radix_cache"):
        cache._store_anchor_kv(req, kv_indices=None)
    # The store should remain empty AND a warning should have been logged
    assert cache.anchor_kv_store == {}
    msgs = [r.message for r in caplog.records if "anchor_kv_store" in r.message]
    assert any("missing code_anchor_token_spans" in m for m in msgs), msgs


def test_store_anchor_kv_quiet_when_signature_also_missing(monkeypatch, caplog):
    """When even the signature is empty, _store_anchor_kv should still
    silent-return (the request never intended to opt into lossy)."""
    from sglang.srt.mem_cache import radix_cache as rc
    cache = _make_minimal_radix_cache(monkeypatch)

    req = SimpleNamespace(
        rid="test-rid-2",
        code_anchor_signature="",
        code_content_signature="",
        code_anchor_token_spans=[],
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[],
    )
    with caplog.at_level(logging.WARNING, logger="sglang.srt.mem_cache.radix_cache"):
        cache._store_anchor_kv(req, kv_indices=None)
    assert cache.anchor_kv_store == {}
    # No warning expected — the request never opted in.
    assert not any("anchor_kv_store" in r.message for r in caplog.records)
