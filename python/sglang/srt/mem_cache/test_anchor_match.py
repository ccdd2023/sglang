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
import time
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
    cache.protected_size_ = 0
    cache.evictable_leaves = set()
    cache.root_node = rc.TreeNode(priority=0)
    cache.device = torch.device("cpu")
    cache.disable = False
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


def test_decrement_anchor_refs_releases_prefetch_lock(monkeypatch):
    """Protected AgentTemplateKV anchors must release their radix lock before
    GC drops the entry."""
    from sglang.srt.mem_cache import radix_cache as rc
    cache = _make_minimal_radix_cache(monkeypatch)
    source_node = _make_source_node(rc, cache, [1, 2, 3])

    entry = rc.AnchorKVEntry(
        signature="aid-1",
        token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
        kv_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        start_pos=0,
        code_content_signature="content-1",
        source_node=source_node,
    )
    entry.ref_count = 1
    cache.anchor_kv_store["content-1"] = [entry]
    cache._agenttemplatekv_protect_entry(entry)
    assert source_node.lock_ref == 1

    node = _make_tree_node(rc, content_sig="content-1")
    cache._decrement_anchor_refs(node)
    assert source_node.lock_ref == 0
    assert "content-1" not in cache.anchor_kv_store


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


def _make_source_node(rc, cache, tokens):
    node = rc.TreeNode(priority=0)
    node.key = rc.RadixKey(tokens)
    node.value = torch.arange(len(tokens), dtype=torch.int64)
    node.parent = cache.root_node
    cache.root_node.children[cache.get_child_key_fn(node.key)] = node
    cache.evictable_size_ = len(tokens)
    return node


def _make_agenttemplatekv_req(**overrides):
    base = dict(
        rid="agenttemplatekv-rid",
        code_anchor_signature="anchor-sig",
        code_content_signature="content-sig",
        code_anchor_token_spans=[
            {"start_token": 1, "end_token": 4, "content_signature": "content-sig"}
        ],
        origin_input_ids=[0, 11, 12, 13, 99],
        output_ids=[],
        codebase_prefetch_hints=[
            {
                "content_signature": "content-sig",
                "text": "abc",
                "steps_to_use": 1,
            }
        ],
        codebase_prefetch_matched_tokens=0,
        codebase_prefetch_success_count=0,
        codebase_prefetch_device_hit_count=0,
        agenttemplatekv_prefetch_hit_count=0,
        agenttemplatekv_prefetch_miss_count=0,
        agenttemplatekv_prefetch_protected_tokens=0,
        agenttemplatekv_prefetch_newly_protected_tokens=0,
        agenttemplatekv_prefetch_consumed_count=0,
        agenttemplatekv_prefetch_expired_tokens=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_agenttemplatekv_store_protects_hint_anchor(monkeypatch):
    """HiCache-independent path: when a finished Planner stores an exact
    codebase anchor listed in its hints, AgentTemplateKV pins the source radix
    node on device for the next agent."""
    from sglang.srt.mem_cache import radix_cache as rc

    cache = _make_minimal_radix_cache(monkeypatch)
    source_node = _make_source_node(rc, cache, [0, 11, 12, 13, 99])
    req = _make_agenttemplatekv_req()
    kv_indices = torch.arange(5, dtype=torch.int64)

    cache._store_anchor_kv(req, kv_indices, source_node=source_node)

    entry = cache.anchor_kv_store["content-sig"][0]
    assert entry.prefetch_lock_held is True
    assert source_node.lock_ref == 1
    assert req.codebase_prefetch_device_hit_count == 0
    assert req.agenttemplatekv_prefetch_newly_protected_tokens == 3


def test_agenttemplatekv_prefetch_hits_protected_device_anchor(monkeypatch):
    """A later agent carrying the same exact-content hint should see a device
    hit even when HiCache is disabled."""
    from sglang.srt.mem_cache import radix_cache as rc

    cache = _make_minimal_radix_cache(monkeypatch)
    source_node = _make_source_node(rc, cache, [0, 11, 12, 13, 99])
    entry = rc.AnchorKVEntry(
        signature="anchor-sig",
        token_ids=torch.tensor([11, 12, 13], dtype=torch.int64),
        kv_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        start_pos=1,
        code_content_signature="content-sig",
        source_node=source_node,
    )
    cache.anchor_kv_store["content-sig"] = [entry]

    class Tok:
        def encode(self, text, add_special_tokens=False):
            assert text == "abc"
            return [11, 12, 13]

    req = _make_agenttemplatekv_req()
    cache.agenttemplatekv_prefetch_codebases(req, tokenizer=Tok())

    assert entry.prefetch_lock_held is True
    assert source_node.lock_ref == 1
    assert req.codebase_prefetch_device_hit_count == 1
    assert req.agenttemplatekv_prefetch_hit_count == 1
    assert req.agenttemplatekv_prefetch_protected_tokens == 3


def test_agenttemplatekv_prefetch_ttl_releases_lock(monkeypatch):
    from sglang.srt.mem_cache import radix_cache as rc

    cache = _make_minimal_radix_cache(monkeypatch)
    source_node = _make_source_node(rc, cache, [1, 2, 3])
    entry = rc.AnchorKVEntry(
        signature="anchor-sig",
        token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
        kv_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        start_pos=0,
        code_content_signature="content-sig",
        source_node=source_node,
    )
    cache.anchor_kv_store["content-sig"] = [entry]
    cache._agenttemplatekv_protect_entry(entry, ttl_s=0.001)
    assert source_node.lock_ref == 1

    entry.prefetch_protected_until = time.monotonic() - 1.0
    req = _make_agenttemplatekv_req(codebase_prefetch_hints=[])
    cache._agenttemplatekv_release_expired_prefetch_entries(req)
    assert source_node.lock_ref == 0
    assert req.agenttemplatekv_prefetch_expired_tokens == 3


def test_agenttemplatekv_rejects_large_zero_fill_gap(monkeypatch):
    from sglang.srt.mem_cache import radix_cache as rc

    monkeypatch.setenv("SGLANG_LOSSY_FUZZY_MATCH", "1")
    monkeypatch.setenv("SGLANG_LOSSY_MAX_ZERO_GAP", "4")
    cache = _make_minimal_radix_cache(monkeypatch)
    entry = rc.AnchorKVEntry(
        signature="anchor-sig",
        token_ids=torch.tensor([7, 8, 9], dtype=torch.int64),
        kv_indices=torch.tensor([20, 21, 22], dtype=torch.int64),
        start_pos=20,
        code_content_signature="content-sig",
    )
    cache.anchor_kv_store["content-sig"] = [entry]
    req = SimpleNamespace(
        lossy_first_match_reason="exact_code_content_signature",
        lossy_first_matched_content_signature="content-sig",
        code_anchor_token_spans=[
            {"start_token": 20, "end_token": 23, "content_signature": "content-sig"}
        ],
    )
    key_tokens = [0, 1] + [2] * 18 + [7, 8, 9]
    exact_values = [torch.tensor([100, 101], dtype=torch.int64)]

    values, node = cache._try_lossy_fuzzy_match(
        req,
        rc.RadixKey(key_tokens),
        exact_values,
        exact_node=cache.root_node,
        best_node=cache.root_node,
    )

    assert values == exact_values
    assert node is cache.root_node
    assert req.lossy_rejected_reason == "agenttemplatekv_large_zero_gap"
    assert req.agenttemplatekv_rejected_large_gap_count == 1


def test_master_gate_SGLANG_LOSSY_ENABLED_disables_protect_entry(monkeypatch):
    """When SGLANG_LOSSY_ENABLED=0, _agenttemplatekv_protect_entry is a no-op
    and the source_node's lock_ref is unchanged. Same for
    agenttemplatekv_prefetch_codebases.
    """
    from sglang.srt.mem_cache import radix_cache as rc

    monkeypatch.setenv("SGLANG_LOSSY_ENABLED", "0")
    monkeypatch.setenv("SGLANG_LOSSY_FUZZY_MATCH", "1")
    cache = _make_minimal_radix_cache(monkeypatch)
    source_node = _make_source_node(rc, cache, [1, 2, 3])
    entry = rc.AnchorKVEntry(
        signature="anchor-sig",
        token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
        kv_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        start_pos=0,
        code_content_signature="content-sig",
        source_node=source_node,
    )
    cache.anchor_kv_store["content-sig"] = [entry]
    # protect_entry is a no-op when gate is off
    result = cache._agenttemplatekv_protect_entry(entry)
    assert result is False
    assert source_node.lock_ref == 0
    assert entry.prefetch_lock_held is False
    # Prefetch is also a no-op (returns None)
    req = _make_agenttemplatekv_req(codebase_prefetch_hints=[])
    cache.agenttemplatekv_prefetch_codebases(req)
    assert req.agenttemplatekv_prefetch_protected_tokens == 0
    assert req.agenttemplatekv_prefetch_hit_count == 0


def test_master_gate_SGLANG_LOSSY_ENABLED_default_is_on(monkeypatch):
    """Default SGLANG_LOSSY_ENABLED=1 (omitted env var) — protect_entry fires normally.
    """
    from sglang.srt.mem_cache import radix_cache as rc

    monkeypatch.delenv("SGLANG_LOSSY_ENABLED", raising=False)
    monkeypatch.setenv("SGLANG_LOSSY_FUZZY_MATCH", "1")
    cache = _make_minimal_radix_cache(monkeypatch)
    source_node = _make_source_node(rc, cache, [1, 2, 3])
    entry = rc.AnchorKVEntry(
        signature="anchor-sig",
        token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
        kv_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        start_pos=0,
        code_content_signature="content-sig",
        source_node=source_node,
    )
    cache.anchor_kv_store["content-sig"] = [entry]
    result = cache._agenttemplatekv_protect_entry(entry)
    assert result is True
    assert source_node.lock_ref == 1
    assert entry.prefetch_lock_held is True


def test_protected_size_cap_uses_allocator_size_when_available(monkeypatch):
    """The cap helper prefers self.token_to_kv_pool_allocator.size over the
    SGLANG_MAX_TOTAL_TOKENS env var. Set allocator.size=100000 and verify
    the cap is 50000 (50% of 100000), regardless of the env var.
    """
    from sglang.srt.mem_cache import radix_cache as rc

    monkeypatch.setenv("SGLANG_LOSSY_PROTECTED_FRAC", "0.5")
    monkeypatch.setenv("SGLANG_MAX_TOTAL_TOKENS", "1000")  # would give cap=500

    cache = _make_minimal_radix_cache(monkeypatch)
    # Inject a fake allocator with .size = 100000
    class FakeAllocator:
        size = 100000
    cache.token_to_kv_pool_allocator = FakeAllocator()
    cap = cache._agenttemplatekv_protected_size_cap()
    assert cap == 50000, f"expected 50000 (50% of 100000), got {cap}"


def test_force_evict_oldest_protected_anchor_on_cap_hit(monkeypatch):
    """F3 logic: when protected_size_ is at the cap and a new protect would
    exceed it, _agenttemplatekv_evict_oldest_protected releases the oldest
    still-locked anchor to make room.
    """
    from sglang.srt.mem_cache import radix_cache as rc

    monkeypatch.setenv("SGLANG_LOSSY_FUZZY_MATCH", "1")
    monkeypatch.setenv("SGLANG_LOSSY_PROTECTED_MAX_TOKENS", "10")
    cache = _make_minimal_radix_cache(monkeypatch)

    # Create 2 protected entries, each with 4 tokens, totaling 8 < cap 10
    nodes = []
    entries = []
    for i in range(2):
        n = _make_source_node(rc, cache, [10 + i, 11 + i, 12 + i, 13 + i])
        nodes.append(n)
        e = rc.AnchorKVEntry(
            signature=f"sig-{i}",
            token_ids=torch.tensor([10 + i, 11 + i, 12 + i, 13 + i], dtype=torch.int64),
            kv_indices=torch.tensor([10 + i, 11 + i, 12 + i, 13 + i], dtype=torch.int64),
            start_pos=0,
            code_content_signature=f"content-{i}",
            source_node=n,
        )
        entries.append(e)
        cache.anchor_kv_store[f"content-{i}"] = [e]
        cache._agenttemplatekv_protect_entry(e, ttl_s=60)
    # protected_size_ should be 8 (two 4-token entries)
    assert cache.protected_size_ == 8
    # Add a 3rd entry that would push to 12 > cap 10
    n3 = _make_source_node(rc, cache, [20, 21, 22, 23, 24])
    e3 = rc.AnchorKVEntry(
        signature="sig-2",
        token_ids=torch.tensor([20, 21, 22, 23, 24], dtype=torch.int64),
        kv_indices=torch.tensor([20, 21, 22, 23, 24], dtype=torch.int64),
        start_pos=0,
        code_content_signature="content-2",
        source_node=n3,
    )
    cache.anchor_kv_store["content-2"] = [e3]
    result = cache._agenttemplatekv_protect_entry(e3, ttl_s=60)
    # F3 should have evicted one of the first two entries
    assert result is True
    # protected_size_ should now be <= cap (10)
    assert cache.protected_size_ <= 10
    # The new entry should be locked
    assert e3.prefetch_lock_held is True


def test_env_var_rename_SGLANG_AGENTTEMPLATEKV_is_now_SGLANG_LOSSY(monkeypatch):
    """Verify the env-var rename: the SGLANG_LOSSY_* prefix replaces the
    older SGLANG_AGENTTEMPLATEKV_* prefix. The old name must NOT be read;
    the new name must be read. The legacy SGLANG_LOSSY_FUZZY_MATCH=1
    enables the path so the cap test fires.
    """
    import os
    from sglang.srt.mem_cache import radix_cache as rc

    monkeypatch.setenv("SGLANG_LOSSY_FUZZY_MATCH", "1")
    monkeypatch.setenv("SGLANG_LOSSY_PROTECTED_FRAC", "0.5")
    # Old name must NOT be honored
    monkeypatch.setenv("SGLANG_AGENTTEMPLATEKV_PROTECTED_FRAC", "0.99")
    # Set the env var to be ignored (use a very high frac via new name; the
    # allocator.size-based cap should still apply)
    cache = _make_minimal_radix_cache(monkeypatch)
    class FakeAllocator:
        size = 100000
    cache.token_to_kv_pool_allocator = FakeAllocator()
    # The old env var must NOT have been read; the new one is in effect.
    # The new name defaults to 0.5 (50% of 100000 = 50000) regardless of
    # the old-name override.
    cap = cache._agenttemplatekv_protected_size_cap()
    assert cap == 50000, (
        f"old SGLANG_AGENTTEMPLATEKV_PROTECTED_FRAC=0.99 should be ignored; "
        f"new SGLANG_LOSSY_PROTECTED_FRAC=0.5 should give 50% of 100000 = 50000; "
        f"got {cap}"
    )


def test_inc_lock_ref_default_walks_to_root(monkeypatch):
    """Without max_ancestors, inc_lock_ref walks all the way to root_node
    (preserves the original RadixCache contract)."""
    from sglang.srt.mem_cache import radix_cache as rc

    cache = _make_minimal_radix_cache(monkeypatch)
    # Build a 4-level chain root -> a -> b -> c -> leaf
    a = _make_source_node(rc, cache, [1, 2, 3])
    b = rc.TreeNode(priority=0)
    b.key = rc.RadixKey([4, 5, 6])
    b.value = torch.arange(3, 6, dtype=torch.int64)
    b.parent = a
    a.children[cache.get_child_key_fn(b.key)] = b
    leaf = rc.TreeNode(priority=0)
    leaf.key = rc.RadixKey([7, 8, 9])
    leaf.value = torch.arange(6, 9, dtype=torch.int64)
    leaf.parent = b
    b.children[cache.get_child_key_fn(leaf.key)] = leaf
    cache.evictable_size_ += 6  # b + leaf tokens

    result = cache.inc_lock_ref(leaf)
    # Default: walk leaf -> b -> a -> root; lock_ref=1 on all 3
    assert leaf.lock_ref == 1
    assert b.lock_ref == 1
    assert a.lock_ref == 1
    assert result.delta == -9
    # No locked_nodes on default (full-walk path) — caller tracks the leaf
    assert result.locked_nodes is None

    # Symmetric dec_lock_ref also walks to root
    result_dec = cache.dec_lock_ref(leaf)
    assert result_dec.delta == 9
    assert leaf.lock_ref == 0
    assert b.lock_ref == 0
    assert a.lock_ref == 0


def test_inc_lock_ref_capped_returns_locked_nodes(monkeypatch):
    """With max_ancestors=2, inc_lock_ref stops at 2 levels (leaf + 2 ancestors).
    The returned locked_nodes list is used by AgentTemplateKV to release
    exactly those nodes via dec_lock_ref(max_ancestors=...)."""
    from sglang.srt.mem_cache import radix_cache as rc

    cache = _make_minimal_radix_cache(monkeypatch)
    a = _make_source_node(rc, cache, [1, 2, 3])
    b = rc.TreeNode(priority=0)
    b.key = rc.RadixKey([4, 5, 6])
    b.value = torch.arange(3, 6, dtype=torch.int64)
    b.parent = a
    a.children[cache.get_child_key_fn(b.key)] = b
    leaf = rc.TreeNode(priority=0)
    leaf.key = rc.RadixKey([7, 8, 9])
    leaf.value = torch.arange(6, 9, dtype=torch.int64)
    leaf.parent = b
    b.children[cache.get_child_key_fn(leaf.key)] = leaf
    cache.evictable_size_ += 6  # b + leaf

    result = cache.inc_lock_ref(leaf, max_ancestors=2)
    # Capped walk: leaf, b, a — all 3 locked
    assert leaf.lock_ref == 1
    assert b.lock_ref == 1
    assert a.lock_ref == 1
    assert len(result.locked_nodes) == 3
    assert result.locked_nodes[0] is leaf
    assert result.locked_nodes[1] is b
    assert result.locked_nodes[2] is a

    # Release via dec_lock_ref with same cap
    result_dec = cache.dec_lock_ref(leaf, max_ancestors=3)
    assert result_dec.delta == 9
    assert leaf.lock_ref == 0
    assert b.lock_ref == 0
    assert a.lock_ref == 0


def test_alloc_with_defrag_gated_by_env(monkeypatch):
    """Phase 4.3: alloc_with_defrag is gated by SGLANG_KV_ALLOCATOR_DEFRAG.

    With need_sort=False, alloc() does NOT call merge_and_sort_free on its
    own, so the only way to get a contiguous prefix slice when free_pages
    is fragmented is to call alloc_with_defrag. With the env var off, the
    prefix slice stays fragmented; with the env var on, merge_and_sort_free
    runs first and the prefix slice is contiguous.
    """
    from sglang.srt.mem_cache import allocator as alloc_mod

    a = alloc_mod.TokenToKVPoolAllocator.__new__(alloc_mod.TokenToKVPoolAllocator)
    a.device = torch.device("cpu")
    a.size = 100
    a.need_sort = False  # critical: alloc() won't defrag on its own
    # Fragmented free list: small head, material scattered in release_pages
    a.free_pages = torch.tensor([50, 51, 52], dtype=torch.int64)  # head too small
    a.release_pages = torch.tensor(
        list(range(10, 30)), dtype=torch.int64
    )  # 20 indices below
    a.is_not_in_free_group = True
    a.free_group = []

    # Without env var: alloc() returns just the head (3 tokens) — not None,
    # but NOT the requested 20. This simulates the failure mode common.py
    # used to retry on: the alloc "succeeds" with a too-small slice.
    monkeypatch.delenv("SGLANG_KV_ALLOCATOR_DEFRAG", raising=False)
    out = a.alloc_with_defrag(20)
    # Without defrag, alloc returns 3 of 20 → 20 is NOT satisfied.
    # Real allocator.alloc would return None here; alloc_with_defrag just
    # delegates to alloc() with no extra defrag, so we get None when
    # need_size > free_pages. Verify the state: free_pages and release_pages
    # are still fragmented (no merge happened).
    assert out is None  # need_size 20 > free_pages 3
    assert len(a.free_pages) == 3
    assert len(a.release_pages) == 20

    # With env var: merge_and_sort_free runs first. free_pages becomes the
    # sorted concatenation [10,11,...,29,50,51,52] (23 entries). The prefix
    # slice of 20 is contiguous.
    monkeypatch.setenv("SGLANG_KV_ALLOCATOR_DEFRAG", "1")
    out = a.alloc_with_defrag(20)
    assert out is not None
    assert out.numel() == 20
    diffs = out[1:] - out[:-1]
    assert (diffs == 1).all(), f"alloc returned non-contiguous slice: {out}"


def test_agenttemplatekv_cache_subclass_dispatch(monkeypatch):
    """Phase 4.4: AgentTemplateKVCache is a RadixCache subclass that exposes
    prefetch_codebases. A stock RadixCache is NOT a subclass, so the
    scheduler's isinstance check is the upstreamable dispatch."""
    from sglang.srt.mem_cache.agenttemplatekv_cache import AgentTemplateKVCache
    from sglang.srt.mem_cache import radix_cache as rc

    # Stock RadixCache (created via __new__ to skip __init__) is NOT an
    # instance of AgentTemplateKVCache.
    cache = rc.RadixCache.__new__(rc.RadixCache)
    assert not isinstance(cache, AgentTemplateKVCache)
    # The agenttemplatekv_prefetch_codebases alias on the base class still
    # exists for back-compat with older scheduler call sites.
    assert hasattr(cache, "agenttemplatekv_prefetch_codebases")

    # AgentTemplateKVCache IS a RadixCache (subclass relationship holds)
    assert issubclass(AgentTemplateKVCache, rc.RadixCache)
    # And it has the public prefetch_codebases method
    assert hasattr(AgentTemplateKVCache, "prefetch_codebases")
    # The two methods are the same callable
    assert (
        AgentTemplateKVCache.prefetch_codebases
        is not rc.RadixCache.agenttemplatekv_prefetch_codebases
    )


# ---------------------------------------------------------------------------
# Regression tests for force-evict path (2026-06-09)
# See results/pass100_attempt/REPORT.md Step 2.10 for context.
# ---------------------------------------------------------------------------


class _MockAllocator:
    """Mock allocator that just records what was freed."""

    def __init__(self):
        self.freed_values = []

    def free(self, value):
        self.freed_values.append(value)


class _LRUStrategy:
    """Mock eviction strategy that returns the node's id() as priority
    (so sort order is deterministic by insertion)."""

    def get_priority(self, node):
        # Lower priority = evicted first (heapq is a min-heap)
        return node._priority_marker


def _build_force_evict_setup(monkeypatch):
    """Build a minimal RadixCache + tree with 3 leaves: 2 locked at r=3,
    1 unlocked. The 2 locked leaves simulate the 8K-prefill OOM
    scenario from results/pass100_attempt/REPORT.md Step 2.4."""
    from sglang.srt.mem_cache import radix_cache as rc
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    cache = rc.RadixCache.__new__(rc.RadixCache)
    cache.get_child_key_fn = rc.get_child_key
    cache.page_size = 1
    cache.anchor_kv_store = {}
    cache.anchor_kv_store_lock = threading.RLock()
    cache.evictable_size_ = 0
    cache.protected_size_ = 0
    cache.evictable_leaves = set()
    cache.root_node = rc.TreeNode(priority=0)
    cache.device = torch.device("cpu")
    cache.disable = False
    cache.eviction_strategy = _LRUStrategy()
    cache.token_to_kv_pool_allocator = _MockAllocator()
    cache.enable_kv_cache_events = False  # skip _record_remove_event path

    # Build 3 leaves under the root:
    #   - leaf_a (locked r=3, 8K tokens) — typical OOM victim
    #   - leaf_b (locked r=3, 8K tokens) — typical OOM victim
    #   - leaf_c (r=0, 4K tokens) — would already be evictable normally
    counter = [0]

    def add_leaf(num_tokens, lock_ref):
        counter[0] += 1
        n = rc.TreeNode(priority=0)
        n.key = rc.RadixKey([1000 + counter[0]] * num_tokens)
        n.value = torch.arange(num_tokens, dtype=torch.int64)
        n.parent = cache.root_node
        n._priority_marker = counter[0]  # for LRU strategy
        n.lock_ref = lock_ref
        cache.root_node.children[cache.get_child_key_fn(n.key)] = n
        if lock_ref == 0:
            cache.evictable_leaves.add(n)
            cache.evictable_size_ += num_tokens
        else:
            cache.protected_size_ += num_tokens
        return n

    leaf_a = add_leaf(8192, lock_ref=3)
    leaf_b = add_leaf(8192, lock_ref=3)
    leaf_c = add_leaf(4096, lock_ref=0)

    return cache, leaf_a, leaf_b, leaf_c, EvictParams


def test_force_evict_bypasses_lock_ref(monkeypatch):
    """force_evict() must free leaves whose lock_ref > 0, the OOM
    recovery scenario documented in results/pass100_attempt/REPORT.md
    Step 2.10."""
    cache, leaf_a, leaf_b, leaf_c, EvictParams = _build_force_evict_setup(monkeypatch)

    # Sanity: normal evict() should free ONLY leaf_c (4096 tokens),
    # because leaf_a and leaf_b are r=3 and not in evictable_leaves.
    result_normal = cache.evict(EvictParams(num_tokens=16384))
    assert result_normal.num_tokens_evicted == 4096, (
        f"normal evict should have freed only leaf_c (4096), got {result_normal.num_tokens_evicted}"
    )
    assert len(cache.token_to_kv_pool_allocator.freed_values) == 1
    assert cache.token_to_kv_pool_allocator.freed_values[0].numel() == 4096
    # Reset the allocator's record so the next phase starts clean
    cache.token_to_kv_pool_allocator.freed_values.clear()

    # Re-attach leaf_a and leaf_b for the force test (the prior evict
    # removed leaf_c but leaf_a and leaf_b are still in the tree).
    cache.evictable_size_ = 16384  # 2x 8192 (the locked ones are still in tree)
    cache.protected_size_ = 16384
    # evictable_leaves should still be empty (no r=0 leaves)
    assert len(cache.evictable_leaves) == 0, (
        "precondition: 2 r=3 leaves should not be in evictable_leaves"
    )

    # Now force evict: should free the 2 locked leaves (16384 tokens)
    result_force = cache.evict(EvictParams(num_tokens=16384, force=True))
    assert result_force.num_tokens_evicted == 16384, (
        f"force evict should have freed 16384 (2x 8192), got {result_force.num_tokens_evicted}"
    )
    # Mock allocator should have recorded 2 frees from the force path
    freed = cache.token_to_kv_pool_allocator.freed_values
    assert len(freed) == 2
    assert all(v.numel() == 8192 for v in freed)


def test_force_evict_marks_leaves_evicted(monkeypatch):
    """force_evict() must set node.evicted=True on freed leaves so a
    later dec_lock_ref from the in-flight request does not try to
    re-add them to evictable_leaves."""
    cache, leaf_a, leaf_b, leaf_c, EvictParams = _build_force_evict_setup(monkeypatch)
    # Force-evict everything
    cache.evict(EvictParams(num_tokens=100000, force=True))
    # The freed leaves (a, b, c) should all have evicted=True now
    assert leaf_a.evicted is True
    assert leaf_b.evicted is True
    assert leaf_c.evicted is True


def test_force_evict_respects_num_tokens(monkeypatch):
    """force_evict() should stop freeing once num_tokens is satisfied,
    not blindly free everything in the tree."""
    cache, leaf_a, leaf_b, leaf_c, EvictParams = _build_force_evict_setup(monkeypatch)
    # Request only 8K (one leaf)
    result = cache.evict(EvictParams(num_tokens=8192, force=True))
    assert result.num_tokens_evicted == 8192
    # Should have freed exactly 1 leaf
    assert len(cache.token_to_kv_pool_allocator.freed_values) == 1


def test_normal_evict_does_not_force(monkeypatch):
    """The default path (force=False) must NOT bypass lock_ref — this
    is the upstream SGLang behavior we preserve by default."""
    cache, leaf_a, leaf_b, leaf_c, EvictParams = _build_force_evict_setup(monkeypatch)
    # Normal evict: should free only leaf_c (the only one in evictable_leaves)
    result = cache.evict(EvictParams(num_tokens=100000))
    assert result.num_tokens_evicted == 4096
    # leaf_a and leaf_b (r=3) should still be in the tree
    assert cache.root_node.children  # still has children
    # leaf_c should be gone
    assert not any(
        c.value.numel() == 4096 for c in cache.root_node.children.values()
    ), "leaf_c should have been evicted by normal path"

