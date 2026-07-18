from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmark.multi_workflow.analyze_probehead_v12 import (
    calibrate,
    gate_composition,
)
from benchmark.multi_workflow.measure_probehead_v12 import (
    _probe_metrics,
    plan_rows,
)
from benchmark.multi_workflow.measure_sessiongraph_atlas import _rope_shift
from benchmark.multi_workflow.prepare_probehead_v12 import prepare
from benchmark.multi_workflow.probehead_v12 import (
    HEAD_CANDIDATES,
    PROFILE,
    ProbeCandidate,
    decide_probe_candidates,
    probe_score,
    shuffled_exact_budget,
)
from benchmark.multi_workflow.sessiongraph_v11 import CostModel


def _cost() -> CostModel:
    return CostModel(
        dense_us_per_token=10.0,
        copy_us_per_token=1.0,
        rope_us_per_token=0.0,
        island_fixed_us=1.0,
        cpu_lookup_us=1.0,
        safety_margin_us=1.0,
    )


def _candidate(module: str, start: int, length: int = 40) -> ProbeCandidate:
    return ProbeCandidate(
        session_id="s",
        turn_id=1,
        module_id=module,
        source_start=start,
        target_start=start,
        length=length,
        prompt_tokens=200,
    )


def _jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_probe_score_and_online_island_decisions_fail_closed():
    assert probe_score(0.1, 0.2) == 0.2
    with pytest.raises(ValueError):
        probe_score(float("nan"), 0.0)
    candidates = [
        _candidate("a", 0),
        _candidate("b", 40),
        _candidate("c", 80),
    ]
    decisions = decide_probe_candidates(
        candidates=candidates,
        scores={"a": 0.01, "b": 0.9, "c": 0.02},
        head_tokens=8,
        threshold=0.1,
        cost_model=_cost(),
    )
    assert [row.reason for row in decisions] == [
        "probe_copy",
        "probe_score_above_threshold",
        "probe_copy",
    ]
    assert [row.island_index for row in decisions] == [0, None, 1]
    assert sum(row.copied_tokens for row in decisions) == 64


def test_shuffled_control_matches_exact_body_budget():
    candidates = [_candidate(f"m{index}", index * 40) for index in range(4)]
    rows = shuffled_exact_budget(
        candidates=candidates,
        copied_token_budget=70,
        head_tokens=8,
    )
    assert sum(row.copied_tokens for row in rows) == 70
    assert all(0 <= row.copied_tokens <= 32 for row in rows)
    assert any(row.head_tokens not in {8, 40} for row in rows)


def test_probe_metric_applies_rope_to_k_but_not_v():
    source_key = torch.randn(2, 8, 4)
    source_value = torch.randn(2, 8, 4)
    target_key = torch.zeros(2, 10, 4)
    target_value = torch.zeros(2, 10, 4)
    target_key[:, 3:5] = _rope_shift(source_key[:, 1:3], 2, 10_000.0)
    target_value[:, 3:5] = source_value[:, 1:3]
    k_dev, v_dev, elapsed = _probe_metrics(
        source_cache=[(source_key, source_value)],
        target_cache=[(target_key, target_value)],
        source_span=(1, 3),
        target_span=(3, 5),
        head_tokens=2,
        theta=10_000.0,
    )
    assert k_dev == pytest.approx(0.0, abs=1e-6)
    assert v_dev == pytest.approx(0.0, abs=1e-6)
    assert elapsed >= 0


def test_prepare_freezes_v11_inputs_without_model_outputs(tmp_path: Path):
    replay = []
    capacity = []
    development = [f"dev-{index:02d}" for index in range(32)]
    holdout = [f"hold-{index:02d}" for index in range(32)]
    for session in [*development, *holdout]:
        base = {
            "module_id": "m",
            "module_type": "source_view",
            "cache_scope": "workspace",
            "content_hash": "same",
            "token_span": [5, 45],
        }
        replay.extend(
            [
                {
                    "session_id": session,
                    "turn_id": 0,
                    "modules": [base],
                },
                {
                    "session_id": session,
                    "turn_id": 1,
                    "modules": [base],
                },
                {
                    "session_id": session,
                    "turn_id": 2,
                    "modules": [base],
                },
                {
                    "session_id": session,
                    "turn_id": 3,
                    "modules": [base],
                },
            ]
        )
        for turn in (1, 2, 3):
            capacity.append(
                {
                    "session_id": session,
                    "turn_id": turn,
                    "prompt_tokens": 100,
                    "copied_module_ids": ["m"],
                }
            )
    replay_path = tmp_path / "replay.json"
    capacity_path = tmp_path / "capacity.jsonl"
    split_path = tmp_path / "split.json"
    v11_design = tmp_path / "v11-design.jsonl"
    v11_registration = tmp_path / "v11-registration.json"
    replay_path.write_text(json.dumps(replay))
    _jsonl(capacity_path, capacity)
    split_path.write_text(
        json.dumps({"development": development, "holdout": holdout})
    )
    _jsonl(
        v11_design,
        [
            {
                "cohort": "development",
                "session_id": development[0],
                "turn_id": 3,
                "module_id": "m",
                "module_type": "source_view",
                "cache_scope": "workspace",
                "disturbance": "identity",
                "negative_control": True,
                "recompute_fraction": 0.0,
            }
        ],
    )
    v11_registration.write_text(
        json.dumps({"policy": "fileversion-sessiongraphkv-v11"})
    )
    result = prepare(
        replay_path=replay_path,
        capacity_path=capacity_path,
        split_path=split_path,
        v11_design_path=v11_design,
        v11_registration_path=v11_registration,
        output_dir=tmp_path / "v12",
    )
    registration = json.loads(
        (tmp_path / "v12/EXPERIMENT_REGISTRATION.json").read_text()
    )
    assert result["workflow_rows"] == 192 * len(HEAD_CANDIDATES)
    assert registration["policy"] == PROFILE
    assert registration["holdout_measurements_read"] is False
    assert registration["v11_thresholds_changed"] is False


def _registration_fixture(tmp_path: Path):
    design = tmp_path / "design.jsonl"
    _jsonl(
        design,
        [
            {
                "cohort": cohort,
                "case_kind": "workflow",
                "session_id": f"{cohort}-s",
                "turn_id": 1,
                "module_id": "m",
                "disturbance": "same_task",
                "head_tokens": head,
            }
            for cohort in ("development", "holdout")
            for head in HEAD_CANDIDATES
        ],
    )
    import hashlib

    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "policy": PROFILE,
                "design_sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
                "gates": {},
                "calibration_rule": "frozen",
            }
        )
    )
    return design, registration


def _amendment(tmp_path: Path) -> Path:
    path = tmp_path / "executor-amendment.json"
    path.write_text(
        json.dumps(
            {
                "accepted": True,
                "probe_warmup_iterations": 3,
                "timed_scope": "vectorized KV comparison only",
                "thresholds_changed": False,
                "holdout_opened": False,
            }
        )
    )
    return path


def test_holdout_plan_requires_calibration_and_passed_development_gate(
    tmp_path: Path,
):
    design, registration = _registration_fixture(tmp_path)
    with pytest.raises(ValueError, match="CALIBRATION_LOCK"):
        plan_rows(
            design_path=design,
            cohort="holdout",
            mode="probes",
            calibration_lock_path=None,
        )
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"status": "LOCKED", "head_tokens": 8, "threshold": 0.1})
    )
    with pytest.raises(ValueError, match="development compose"):
        plan_rows(
            design_path=design,
            cohort="holdout",
            mode="probes",
            calibration_lock_path=lock,
        )
    import hashlib

    gate = tmp_path / "development-gate.json"
    gate.write_text(
        json.dumps(
            {
                "stage": "development-compose",
                "passed": True,
                "inputs": {
                    "calibration_lock": hashlib.sha256(lock.read_bytes()).hexdigest()
                },
            }
        )
    )
    rows, head, threshold = plan_rows(
        design_path=design,
        cohort="holdout",
        mode="probes",
        calibration_lock_path=lock,
        development_gate_path=gate,
    )
    assert len(rows) == 1
    assert (head, threshold) == (8, 0.1)
    assert registration.exists()


def test_calibration_is_deterministic_and_prefers_capacity(tmp_path: Path):
    design, registration = _registration_fixture(tmp_path)
    observations = []
    for head in HEAD_CANDIDATES:
        for module, score, js, start in (
            ("safe", 0.01, 1e-5, 0),
            ("harmful", 0.9, 1e-2, 40),
        ):
            observations.append(
                {
                    "case_kind": "workflow",
                    "cohort": "development",
                    "session_id": "development-s",
                    "turn_id": 1,
                    "module_id": module,
                    "disturbance": "same_task",
                    "head_tokens": head,
                    "status": "ok",
                    "source_start": start,
                    "target_start": start,
                    "token_count": 40,
                    "target_prompt_tokens": 100,
                    "probe_score": score,
                    "probe_ms": 0.001,
                    "causal_splice_logit_js": js,
                    "splice_top1_changed": 0,
                }
            )
    observations_path = tmp_path / "observations.jsonl"
    _jsonl(observations_path, observations)
    _jsonl(
        design,
        [
            {
                key: row[key]
                for key in (
                    "case_kind",
                    "cohort",
                    "session_id",
                    "turn_id",
                    "module_id",
                    "disturbance",
                    "head_tokens",
                )
            }
            for row in observations
        ],
    )
    import hashlib

    registration_value = json.loads(registration.read_text())
    registration_value["design_sha256"] = hashlib.sha256(
        design.read_bytes()
    ).hexdigest()
    registration.write_text(json.dumps(registration_value))
    cost_gate = tmp_path / "cost.json"
    cost_gate.write_text(json.dumps({"cost_model": _cost().__dict__}))
    result = calibrate(
        observations_path=observations_path,
        design_path=design,
        registration_path=registration,
        executor_amendment_path=_amendment(tmp_path),
        cost_gate_path=cost_gate,
        lock_output=tmp_path / "lock.json",
        report_output=tmp_path / "report.json",
    )
    assert result["passed"]
    assert result["chosen"]["head_tokens"] == 8
    assert result["chosen"]["threshold"] == 0.01


def test_composition_gate_uses_session_clustered_controls(tmp_path: Path):
    design, registration = _registration_fixture(tmp_path)
    import hashlib

    amendment = _amendment(tmp_path)
    modules = []
    requests = []
    for session in range(32):
        session_id = f"dev-{session:02d}"
        modules.extend(
            [
                {
                    "status": "ok",
                    "cohort": "development",
                    "case_kind": "stress",
                    "session_id": session_id,
                    "turn_id": 3,
                    "module_id": "stress",
                    "disturbance": disturbance,
                    "head_tokens": 8,
                    "causal_splice_logit_js": 1e-6,
                    "probe_ms": 0.01,
                }
                for disturbance in ("identity", "change_after")
            ]
        )
        modules.append(
            {
                "status": "ok",
                "cohort": "development",
                "case_kind": "workflow",
                "session_id": session_id,
                "turn_id": 1,
                "module_id": "workflow",
                "disturbance": "same_task",
                "head_tokens": 8,
                "causal_splice_logit_js": 1e-5,
                "probe_ms": 0.01,
            }
        )
        for turn in (1, 2, 3):
            requests.append(
                {
                    "status": "ok",
                    "cohort": "development",
                    "session_id": session_id,
                    "turn_id": turn,
                    "cost_positive_copy_fraction": 0.2,
                    "probe_composed_js": 1e-5,
                    "copy_all_composed_js": 1e-3,
                    "shuffled_composed_js": 1e-3,
                    "probe_top1_changed": 0,
                    "probe_p95_ms": 0.01,
                }
            )
    module_path = tmp_path / "modules.jsonl"
    request_path = tmp_path / "requests.jsonl"
    _jsonl(module_path, modules)
    _jsonl(request_path, requests)
    _jsonl(
        design,
        [
            {
                key: row[key]
                for key in (
                    "case_kind",
                    "cohort",
                    "session_id",
                    "turn_id",
                    "module_id",
                    "disturbance",
                    "head_tokens",
                )
            }
            for row in modules
        ],
    )
    registration_value = json.loads(registration.read_text())
    registration_value["design_sha256"] = hashlib.sha256(
        design.read_bytes()
    ).hexdigest()
    registration.write_text(json.dumps(registration_value))
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "head_tokens": 8,
                "threshold": 0.1,
                "registration_sha256": hashlib.sha256(
                    registration.read_bytes()
                ).hexdigest(),
                "executor_amendment_sha256": hashlib.sha256(
                    amendment.read_bytes()
                ).hexdigest(),
            }
        )
    )
    result = gate_composition(
        stage="development-compose",
        module_observations_path=module_path,
        request_observations_path=request_path,
        design_path=design,
        registration_path=registration,
        calibration_lock_path=lock,
        executor_amendment_path=amendment,
        output_path=tmp_path / "gate.json",
        verdict_path=tmp_path / "verdict.md",
        iterations=100,
    )
    assert result["passed"]
    assert result["copy_all_harm_reduction"] == pytest.approx(0.99)
    assert result["shuffled_harm_reduction_ci_low"] > 0
