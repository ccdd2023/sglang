import copy
import json

import pytest

from benchmark.multi_workflow.fair_sota_comparison_v2 import (
    canonical_sha256,
    choose_operating_point,
    discordant_task_ids,
    paired_geometric_speedup,
    select_hash_holdout,
    token_identity_audit,
    validate_ledger,
    validate_workload,
)


def _population():
    return [
        {"instance_id": "alpha__one-1", "problem_statement": "a"},
        {"instance_id": "alpha__one-2", "problem_statement": "b"},
        {"instance_id": "alpha__one-3", "problem_statement": "c"},
        {"instance_id": "beta__two-1", "problem_statement": "d"},
        {"instance_id": "gamma__three-1", "problem_statement": "e"},
    ]


def _case(case_id="case-a"):
    messages = [
        {"role": "system", "content": "complete code"},
        {"role": "user", "content": "def f():\n    return "},
    ]
    return {
        "case_id": case_id,
        "max_new_tokens": 8,
        "messages": messages,
        "prompt_sha256": canonical_sha256(messages),
        "segments": [
            {
                "segment_id": f"{case_id}:context",
                "reusable": True,
                "text": "def f():\n",
            },
            {
                "segment_id": f"{case_id}:query",
                "reusable": False,
                "text": "    return ",
            },
        ],
    }


def _workload():
    return {
        "schema_version": 1,
        "dataset": "repobench-p",
        "cases": [_case("a"), _case("b")],
    }


def _record(case_id, method="v40", mode="reuse", token_hash="same"):
    return {
        "case_id": case_id,
        "config_id": "cap-4096",
        "engine": "sglang",
        "error": None,
        "method": method,
        "mode": mode,
        "prompt_sha256": _case(case_id)["prompt_sha256"],
        "token_ids_sha256": token_hash,
        "ttft_ms": 10.0,
        "reused_k_tokens": 32 if mode == "reuse" else 0,
        "reused_v_tokens": 32 if mode == "reuse" else 0,
        "fallback_reason": None,
        "metadata": {"warmup": False, "source_observation": False},
    }


def test_hash_holdout_is_deterministic_disjoint_and_repository_capped():
    first = select_hash_holdout(
        _population(),
        excluded_ids={"gamma__three-1"},
        size=3,
        per_repository_cap=2,
        salt="fixed",
    )
    second = select_hash_holdout(
        list(reversed(_population())),
        excluded_ids={"gamma__three-1"},
        size=3,
        per_repository_cap=2,
        salt="fixed",
    )

    assert [row["instance_id"] for row in first] == [
        row["instance_id"] for row in second
    ]
    assert "gamma__three-1" not in {
        row["instance_id"] for row in first
    }
    assert (
        sum(row["instance_id"].startswith("alpha__") for row in first) <= 2
    )


def test_validate_workload_checks_message_hash_and_segment_order():
    result = validate_workload(_workload(), expected_case_ids=["a", "b"])
    assert result["cases"] == 2

    invalid = _workload()
    invalid["cases"][0]["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="manifest hash"):
        validate_workload(invalid)

    invalid = _workload()
    invalid["cases"][0]["segments"] = list(
        reversed(invalid["cases"][0]["segments"])
    )
    with pytest.raises(ValueError, match="segment order"):
        validate_workload(invalid)


def test_validate_ledger_requires_coverage_and_physical_or_fallback():
    workload = _workload()
    rows = [_record("a"), _record("b")]
    result = validate_ledger(
        workload,
        rows,
        expected_method="v40",
        expected_mode="reuse",
    )
    assert result["physical_reuse_records"] == 2

    with pytest.raises(ValueError, match="missing cases"):
        validate_ledger(workload, rows[:1])

    no_reuse = _record("a")
    no_reuse["reused_k_tokens"] = 0
    no_reuse["reused_v_tokens"] = 0
    with pytest.raises(ValueError, match="neither physical tokens"):
        validate_ledger(
            {"dataset": "x", "cases": [_case("a")]},
            [no_reuse],
        )
    no_reuse["fallback_reason"] = "not_eligible"
    assert validate_ledger(
        {"dataset": "x", "cases": [_case("a")]},
        [no_reuse],
    )["fallback_records"] == 1


def test_ledger_ignores_warmups_and_source_observations():
    rows = [_record("a"), _record("b")]
    warmup = copy.deepcopy(rows[0])
    warmup["metadata"]["warmup"] = True
    source = copy.deepcopy(rows[0])
    source["metadata"]["source_observation"] = True

    result = validate_ledger(_workload(), rows + [warmup, source])
    assert result["records"] == 2


def test_token_identity_audit_separates_controlled_and_native_layers():
    same = {
        "v40": [_record("a", "v40", token_hash="h")],
        "cacheblend": [
            _record("a", "cacheblend", token_hash="h")
        ],
        "kvcomm": [_record("a", "kvcomm", token_hash="h")],
    }
    assert token_identity_audit(same)["classification"] == "controlled"

    native = copy.deepcopy(same)
    native["kvcomm"][0]["token_ids_sha256"] = "graph-rewrite"
    result = token_identity_audit(native)
    assert result["classification"] == "native_only"
    assert "a" in result["token_hash_mismatches"]


def test_choose_operating_point_is_accuracy_first_then_n4_speed():
    selected = choose_operating_point(
        [
            {"config_id": "fast", "resolved": 4, "n4_speedup": 2.0},
            {"config_id": "accurate", "resolved": 5, "n4_speedup": 1.1},
            {"config_id": "accurate-fast", "resolved": 5, "n4_speedup": 1.2},
        ]
    )
    assert selected["config_id"] == "accurate-fast"


def test_discordance_and_paired_geometric_speedup():
    outcomes = {
        "same": {"dense": True, "v40": True, "cacheblend": True},
        "different": {"dense": True, "v40": False, "cacheblend": True},
    }
    assert discordant_task_ids(outcomes) == ["different"]
    assert paired_geometric_speedup([20, 45], [10, 20]) == pytest.approx(
        (2 * 2.25) ** 0.5
    )

    with pytest.raises(ValueError, match="equal non-zero"):
        paired_geometric_speedup([1], [])
