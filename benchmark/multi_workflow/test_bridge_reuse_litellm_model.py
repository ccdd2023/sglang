import json
from types import SimpleNamespace

import pytest
from jinja2 import Template
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from benchmark.multi_workflow import bridge_reuse_litellm_model as bridge


class _UnderlyingStream:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _LiteLLMStream:
    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self.completion_stream = _UnderlyingStream()

    def __iter__(self):
        return iter(self._chunks)


class _CharacterTokenizer:
    def encode(self, value, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(ids=[ord(character) for character in value])


def _bare_model() -> bridge.BridgeReuseLitellmModel:
    model = object.__new__(bridge.BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        model_kwargs={}, model_name="test-model", native_backend_url=None
    )
    model._instance_nonce = "test"
    model._session_index = 0
    model._request_index = 1
    model._last_stream_stats = {}
    return model


def test_native_generate_payload_translates_sampling_for_sglang_only() -> None:
    common = dict(
        session_id="session-1",
        request_index=3,
        prompt_text_sha256="prompt-hash",
        input_ids=[1, 2, 3],
        segments=[],
        max_new_tokens=2048,
        temperature=0.0,
        repetition_penalty=1.05,
    )
    sglang = bridge.native_generate_payload(backend="sglang-dense", **common)
    kvcomm = bridge.native_generate_payload(backend="kvcomm-dense", **common)

    assert sglang["sampling_params"] == {
        "max_new_tokens": 2048,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
    }
    assert "sampling_params" not in kvcomm
    assert sglang["input_ids_sha256"] == kvcomm["input_ids_sha256"]


def test_render_message_literal_matches_frozen_qwen25_tool_template() -> None:
    group = [
        {
            "role": "assistant",
            "content": "```bash",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "rg -n \'Thing\' pkg/a.py"}',
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "<returncode>0</returncode>\n<output>class Thing: pass</output>",
        },
    ]
    template_path = (
        bridge.Path(__file__).resolve().parent
        / "qwen2_5_coder_tool_chat_template.jinja"
    )
    model = _bare_model()
    model._chat_template = Template(template_path.read_text())
    model._tokenizer = _CharacterTokenizer()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *group,
    ]
    # Exercise the production normalization path: LiteLLM stores arguments as
    # a JSON string, while the Qwen2.5 template receives the decoded object.
    rendered = model._render_prompt(messages)
    expected = "".join(
        bridge.BridgeReuseLitellmModel._render_message_literal(message)
        for message in group
    )
    assert rendered.count(expected) == 1
    prompt_ids = model._tokenizer.encode(
        rendered, add_special_tokens=False
    ).ids
    assert model._group_token_span(prompt_ids, group) is not None


def test_v33b_state_transition_releases_and_vetoes_current_target() -> None:
    mutation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": (
                                "python -c \"from pathlib import Path; "
                                "Path('pkg/a.py').write_text('x')\""
                            )
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]
    target = {"source_id": "source-1"}

    guarded, releases, decision = bridge.apply_current_target_veto(
        arm="coding_state_transition_target_v33b",
        selected_groups=[mutation],
        target=target,
        releases=[],
    )

    assert guarded is None
    assert releases == ["source-1"]
    assert decision == {
        "target_vetoed": True,
        "target_veto_reasons": ["repository_mutation_command"],
    }


def test_v34_critical_event_vetoes_without_phase_cooldown() -> None:
    mutation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": (
                                "python -c \"open('pkg/a.py', 'w').write('x')\""
                            )
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]
    prior_mutation = list(mutation)

    guarded, releases, decision = bridge.apply_current_target_veto(
        arm="coding_critical_current_target_v34",
        selected_groups=[prior_mutation, mutation],
        target={"source_id": "source-2"},
        releases=[],
    )

    assert guarded is None
    assert releases == ["source-2"]
    assert decision == {
        "target_vetoed": True,
        "target_veto_reasons": ["repository_mutation_command"],
    }


def test_v35b_vetoes_first_validation_target_only() -> None:
    mutation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": (
                                "python -c \"open('pkg/a.py', 'w').write('x')\""
                            )
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]
    validation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": "python -m pytest tests/test_a.py"
                        },
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "<returncode>0</returncode><output>1 passed</output>",
        },
    ]

    guarded, releases, decision = bridge.apply_current_target_veto(
        arm="coding_version_validation_target_v35b",
        selected_groups=[mutation, validation],
        target={"source_id": "source-3"},
        releases=[],
    )
    assert guarded is None
    assert releases == ["source-3"]
    assert decision["target_veto_reasons"] == [
        "first_validation_of_version_before_submit"
    ]

    retained, _, second = bridge.apply_current_target_veto(
        arm="coding_version_validation_target_v35b",
        selected_groups=[mutation, validation, validation],
        target={"source_id": "source-4"},
        releases=[],
    )
    assert retained == {"source_id": "source-4"}
    assert second["target_vetoed"] is False


def test_v37_vetoes_patch_decision_and_recognizes_shell_write() -> None:
    shell_write = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": "cat > /testbed/pkg/a.py <<'EOF'\nx = 1\nEOF"
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]
    validation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "pytest tests/test_a.py"},
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "<returncode>0</returncode><output>1 passed</output>",
        },
    ]
    diff = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "git diff"},
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                "<returncode>0</returncode><output>"
                "diff --git a/pkg/a.py b/pkg/a.py</output>"
            ),
        },
    ]

    _, _, after_validation = bridge.apply_current_target_veto(
        arm="coding_patch_lifecycle_target_v37",
        selected_groups=[shell_write, validation],
        target={"source_id": "source-v"},
        releases=[],
    )
    assert after_validation["target_veto_reasons"] == [
        "first_validation_of_version_before_submit"
    ]

    guarded, releases, after_diff = bridge.apply_current_target_veto(
        arm="coding_patch_lifecycle_target_v37",
        selected_groups=[shell_write, validation, diff],
        target={"source_id": "source-d"},
        releases=[],
    )
    assert guarded is None
    assert releases == ["source-d"]
    assert after_diff["target_veto_reasons"] == [
        "patch_diff_before_submission_decision"
    ]
    retained, source_decision = bridge.select_reuse_groups(
        "coding_patch_lifecycle_target_v37",
        [shell_write, validation, diff],
    )
    assert retained == [validation, diff]
    assert source_decision["mode"] == (
        "patch_lifecycle_target_general_source"
    )


def test_v38_latch_vetoes_target_and_stops_future_source() -> None:
    target = {"source_id": "source-v38"}
    guarded, releases, decision = bridge.apply_current_target_veto(
        arm="coding_commit_phase_dense_v38",
        selected_groups=[],
        target=target,
        releases=[],
        commit_phase_latched=True,
    )
    assert guarded is None
    assert releases == ["source-v38"]
    assert decision["target_veto_reasons"] == [
        "repository_commit_phase_latched"
    ]
    retained, source_decision = bridge.select_reuse_groups(
        "coding_commit_phase_dense_v38",
        [[{"role": "tool", "content": "old"}], [{"role": "tool", "content": "new"}]],
    )
    assert retained == [[{"role": "tool", "content": "new"}]]
    assert source_decision["mode"] == (
        "commit_phase_exploration_general_source"
    )
    model = object.__new__(bridge.BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        reuse_arm="coding_commit_phase_dense_v38",
        rolling_history_groups=6,
    )
    model._commit_phase_latched = True
    source, pending, commit_decision = model._future_source(
        prompt_ids=[],
        selected_groups=[],
    )
    assert source is pending is None
    assert commit_decision["mode"] == "commit_phase_dense_latched"
    assert commit_decision["source_registered"] is False

    model._pending_source = None
    model._last_message_count = 10
    model._session_index = 1
    model._request_index = 7
    model._new_session_if_needed([{"role": "system"}, {"role": "user"}])
    assert model._commit_phase_latched is False


def test_v40_future_source_contains_only_grounded_tool_observation() -> None:
    def read_group(path: str, marker: str):
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {
                                "command": f"sed -n '1,200p' {path}"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    f"{marker} " * 100
                    + "<returncode>0</returncode>"
                ),
            },
        ]

    groups = [
        read_group(f"/testbed/pkg/{index}.py", f"evidence-{index}")
        for index in range(6)
    ]
    model = object.__new__(bridge.BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        reuse_arm="coding_grounded_observation_island_v40",
        rolling_history_groups=6,
        reuse_min_tokens=128,
        reuse_copy_cap=4096,
    )
    model._tokenizer = _CharacterTokenizer()
    model._instance_nonce = "v40"
    model._session_index = 1
    model._request_index = 7
    literal = "".join(
        model._render_message_literal(message)
        for group in groups
        for message in group
    )
    prompt_ids = [1, *model._tokenizer.encode(literal).ids, 2]

    source, pending, decision = model._future_source(
        prompt_ids=prompt_ids,
        selected_groups=groups,
    )

    assert source is not None
    assert pending is not None
    assert decision["mode"] == (
        "grounded_version_valid_observation_island"
    )
    assert decision["assistant_tokens_selected"] == 0
    selected_literal = model._render_message_literal(groups[-1][1])
    assert pending["segment_ids"] == model._tokenizer.encode(
        selected_literal
    ).ids


def _v45_read_group(path: str, symbol: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": f"sed -n '1,240p' {path}"
                        },
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                "<returncode>0</returncode><output>"
                f"def {symbol}():\n    return 1\n"
                + ("# source context\n" * 40)
                + "</output>"
            ),
        },
    ]


def _v45_patch_group(symbol: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": (
                                "apply_patch <<'PATCH'\n"
                                "*** Begin Patch\n"
                                "*** Update File: pkg/latest.py\n"
                                f"@@ def {symbol}():\n"
                                "-    return 1\n"
                                "+    return 2\n"
                                "*** End Patch\n"
                                "PATCH"
                            )
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]


def test_v45_future_source_binds_file_version_evidence() -> None:
    groups = [
        _v45_read_group(f"/testbed/pkg/{index}.py", f"symbol_{index}")
        for index in range(5)
    ] + [_v45_read_group("/testbed/pkg/latest.py", "stable_symbol")]
    model = object.__new__(bridge.BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        reuse_arm="coding_versioned_evidence_guard_v45",
        rolling_history_groups=6,
        reuse_min_tokens=128,
        reuse_copy_cap=4096,
    )
    model._tokenizer = _CharacterTokenizer()
    model._instance_nonce = "v45"
    model._session_index = 1
    model._request_index = 7
    literal = "".join(
        model._render_message_literal(message)
        for group in groups
        for message in group
    )
    prompt_ids = [1, *model._tokenizer.encode(literal).ids, 2]

    source, pending, decision = model._future_source(
        prompt_ids=prompt_ids,
        selected_groups=groups,
    )

    assert source is not None
    assert pending is not None
    assert decision["mode"] == "versioned_grounded_observation_guard_v45"
    assert decision["symbol_relaxation_enabled"] is False
    assert pending["source_paths"] == ["pkg/latest.py"]
    assert pending["source_symbols"] == ["stable_symbol"]
    assert len(pending["source_observation_sha256"]) == 64


def test_v45_abstains_on_v40_selected_pathless_source_without_runner_up() -> None:
    groups = [
        _v45_read_group(f"/testbed/pkg/{index}.py", f"symbol_{index}")
        for index in range(5)
    ]
    groups.append(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "rg stable /testbed/pkg"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "<returncode>0</returncode><output>"
                    + ("unlocalized large observation\n" * 80)
                    + "</output>"
                ),
            },
        ]
    )
    literal = "".join(
        bridge.BridgeReuseLitellmModel._render_message_literal(message)
        for group in groups
        for message in group
    )
    prompt_ids = [
        1,
        *[ord(character) for character in literal],
        2,
    ]

    def planner(arm: str):
        model = object.__new__(bridge.BridgeReuseLitellmModel)
        model.config = SimpleNamespace(
            reuse_arm=arm,
            rolling_history_groups=6,
            reuse_min_tokens=128,
            reuse_copy_cap=4096,
        )
        model._tokenizer = _CharacterTokenizer()
        model._instance_nonce = arm
        model._session_index = 1
        model._request_index = 7
        return model

    v40_source, _, v40_decision = planner(
        "coding_grounded_observation_island_v40"
    )._future_source(prompt_ids=prompt_ids, selected_groups=groups)
    v45_source, v45_pending, v45_decision = planner(
        "coding_versioned_evidence_guard_v45"
    )._future_source(prompt_ids=prompt_ids, selected_groups=groups)

    assert v40_source is not None
    assert v40_decision["selected_group_index"] == 4
    assert v45_source is None
    assert v45_pending is None
    assert v45_decision["selected_group_index"] == 4
    assert v45_decision["skip_reason"] == "selected_observation_unlocalized"


def test_v45_target_rejects_every_new_same_file_write() -> None:
    source_group = _v45_read_group("/testbed/pkg/latest.py", "stable")
    segment_ids = [10, 11, 12]
    pending = {
        "source_id": "v45-source",
        "content_hash": "content",
        "length": len(segment_ids),
        "segment_token_hash": bridge.token_ids_hash(segment_ids),
        "source_prefix_token_hash": "source-prefix",
        "source_prompt_hash": "source-prompt",
        "source_start": 3,
        "segment_ids": segment_ids,
        "source_observation_sha256": (
            bridge.versioned_grounded_observation_candidates([source_group])[1][
                "candidate_evidence"
            ][0]["observation_sha256"]
        ),
        "source_paths": ["pkg/latest.py"],
        "source_symbols": ["stable"],
    }
    target_ids = [1, *segment_ids, 2]

    invalid_model = object.__new__(bridge.BridgeReuseLitellmModel)
    invalid_model.config = SimpleNamespace(
        reuse_arm="coding_versioned_evidence_guard_v45"
    )
    invalid_model._pending_source = dict(pending)
    invalid_model._record_client = lambda row: None
    invalid, releases = invalid_model._target_case(
        target_ids,
        selected_groups=[source_group, _v45_patch_group("stable")],
    )

    assert invalid is None
    assert releases == ["v45-source"]
    assert invalid_model._last_target_evidence_guard["reason"] == (
        "same_file_symbol_overlap"
    )

    valid_model = object.__new__(bridge.BridgeReuseLitellmModel)
    valid_model.config = invalid_model.config
    valid_model._pending_source = dict(pending)
    valid_model._instance_nonce = "v45"
    valid_model._session_index = 1
    valid_model._request_index = 8
    valid_model._record_client = lambda row: None
    valid, releases = valid_model._target_case(
        target_ids,
        selected_groups=[source_group, _v45_patch_group("other")],
    )

    assert valid is None
    assert releases == ["v45-source"]
    assert valid_model._last_target_evidence_guard[
        "target_evidence_valid"
    ] is False
    assert valid_model._last_target_evidence_guard["reason"] == (
        "same_file_symbol_disjoint_not_enabled"
    )


def test_v46_pool_registers_observed_path_and_invalidates_scope_write() -> None:
    read = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": 'find /testbed -name "*.py" | head'
                        },
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                "<returncode>0</returncode><output>"
                + "/testbed/pkg/module.py\n" * 30
                + "</output>"
            ),
        },
    ]
    mutation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": (
                                "python -c \"open('/testbed/other/new.py', "
                                "'w').write('x')\""
                            )
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]
    model = object.__new__(bridge.BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        reuse_arm="coding_observed_path_pool_v46",
        rolling_history_groups=6,
        reuse_min_tokens=128,
        reuse_copy_cap=4096,
    )
    model._tokenizer = _CharacterTokenizer()
    model._instance_nonce = "v46"
    model._session_index = 1
    model._request_index = 2

    read_literal = "".join(
        model._render_message_literal(message) for message in read
    )
    source_prompt = [ord(value) for value in "P" + read_literal + "S"]
    sources, pool, releases, decision = model._v46_future_sources(
        prompt_ids=source_prompt,
        selected_groups=[read],
        pool={},
    )

    assert releases == []
    assert len(sources) == len(pool) == 1
    assert sources[0]["persistent"] is True
    assert decision["observation_path_candidates"] == 1

    model._request_index = 3
    target_prompt = [ord(value) for value in "Q" + read_literal + "T"]
    model._pending_sources = pool
    cases, releases, guards, retained = model._v46_target_cases(
        prompt_ids=target_prompt,
        selected_groups=[read],
    )
    assert len(cases) == 1
    assert releases == []
    assert len(retained) == 1
    assert cases[0]["target_group_id"]
    assert guards[0]["target_evidence_valid"] is True

    mutation_literal = "".join(
        model._render_message_literal(message) for message in mutation
    )
    invalid_prompt = [
        ord(value) for value in "Q" + read_literal + mutation_literal + "T"
    ]
    cases, releases, guards, retained = model._v46_target_cases(
        prompt_ids=invalid_prompt,
        selected_groups=[read, mutation],
    )
    assert cases == []
    assert releases == [sources[0]["source_id"]]
    assert retained == {}
    assert guards[0]["reason"] == "repository_scope_mutated"


def test_v46_does_not_evict_sources_referenced_by_current_target() -> None:
    read = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "cat /testbed/pkg/new.py"},
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                "<returncode>0</returncode><output>"
                + "def new_value():\n    return 1\n" * 20
                + "</output>"
            ),
        },
    ]
    model = object.__new__(bridge.BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        reuse_arm="coding_observed_path_pool_v46",
        rolling_history_groups=6,
        reuse_min_tokens=128,
        reuse_copy_cap=4096,
    )
    model._tokenizer = _CharacterTokenizer()
    model._instance_nonce = "v46"
    model._session_index = 1
    model._request_index = 7
    literal = "".join(
        model._render_message_literal(message) for message in read
    )
    prompt_ids = [ord(value) for value in "P" + literal + "S"]
    pool = {
        f"old-{index}": {
            "source_id": f"source-{index}",
            "segment_ids": [index + 1] * (256 + index),
            "source_request_index": index + 1,
        }
        for index in range(3)
    }

    sources, retained, releases, decision = model._v46_future_sources(
        prompt_ids=prompt_ids,
        selected_groups=[read],
        pool=pool,
        protected_source_ids={
            "source-0",
            "source-1",
            "source-2",
        },
    )

    assert sources == []
    assert retained == pool
    assert releases == []
    assert decision["target_protected_sources"] == 3


def test_natural_code_cost_arm_defers_then_admits_valid_source() -> None:
    read = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": "sed -n '1,240p' /testbed/pkg/module.py"
                        },
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                "<returncode>0</returncode><output>"
                + "def value():\n    return 1\n" * 30
                + "</output>"
            ),
        },
    ]
    model = object.__new__(bridge.BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        reuse_arm="coding_natural_code_cost",
        rolling_history_groups=6,
        reuse_min_tokens=128,
        reuse_copy_cap=4096,
    )
    model._tokenizer = _CharacterTokenizer()
    model._instance_nonce = "natural-code"
    model._session_index = 1
    model._request_index = 2

    literal = "".join(
        model._render_message_literal(message) for message in read
    )
    source_prompt = [ord(value) for value in "P" + literal + "S"]
    sources, pool, releases, decision = model._v46_future_sources(
        prompt_ids=source_prompt,
        selected_groups=[read],
        pool={},
    )

    assert releases == []
    assert len(sources) == len(pool) == 1
    assert decision["selected_module_type"] == "repository_code"

    model._pending_sources = pool
    short_prompt = [ord(value) for value in "Q" + literal + "T"]
    cases, releases, guards, retained = model._v46_target_cases(
        prompt_ids=short_prompt,
        selected_groups=[read],
    )
    assert cases == []
    assert releases == []
    assert retained == pool
    assert guards[0]["admission_reason"] == (
        "predicted_cache_ready_saving_nonpositive"
    )

    long_prompt = [ord("Q")] * 2_000 + [
        ord(value) for value in literal + "T"
    ]
    cases, releases, guards, retained = model._v46_target_cases(
        prompt_ids=long_prompt,
        selected_groups=[read],
    )
    assert len(cases) == 1
    assert releases == []
    assert retained == pool
    assert guards[0]["admission_reason"] == (
        "predicted_cache_ready_saving_positive"
    )
    assert cases[0]["cost_estimate"]["reuse_admitted"] is True


def test_dependency_graph_lcb_arm_admits_only_one_best_island() -> None:
    def read(path: str, symbol: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {
                                "command": f"cat /testbed/{path}"
                            },
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "<returncode>0</returncode><output>"
                    + f"def {symbol}():\n    return 1\n" * 45
                    + "</output>"
                ),
            },
        ]

    first = read("pkg/first.py", "first_value")
    second = read("pkg/second.py", "second_value")
    groups = [first, second]
    model = object.__new__(bridge.BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        reuse_arm="coding_dependency_graph_cold_lcb",
        rolling_history_groups=6,
        reuse_min_tokens=128,
        reuse_copy_cap=4096,
    )
    model._tokenizer = _CharacterTokenizer()
    model._instance_nonce = "graph-lcb"
    model._session_index = 1
    model._request_index = 2

    literal = "".join(
        model._render_message_literal(message)
        for group in groups
        for message in group
    )
    source_prompt = [ord(value) for value in "P" + literal + "S"]
    sources, pool, releases, decision = model._v46_future_sources(
        prompt_ids=source_prompt,
        selected_groups=groups,
        pool={},
    )
    assert releases == []
    assert len(sources) == len(pool) == 2
    assert decision["max_target_islands"] == 1
    assert all("source_dependency_graph" in row for row in pool.values())

    model._pending_sources = pool
    target_prompt = [ord("Q")] * 10_000 + [
        ord(value) for value in literal + "T"
    ]
    cases, releases, guards, retained = model._v46_target_cases(
        prompt_ids=target_prompt,
        selected_groups=groups,
    )
    assert releases == []
    assert retained == pool
    assert len(cases) == 1
    assert len(guards) == 2
    assert cases[0]["cost_estimate"][
        "lower_bound_cache_ready_saving_ms"
    ] > 0
    assert all(
        guard["admission_reason"]
        == "lower_bound_cache_ready_saving_positive"
        for guard in guards
    )


def test_query_closes_underlying_sync_stream(monkeypatch) -> None:
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="done",
                    tool_calls=None,
                    reasoning_content=None,
                )
            )
        ]
    )
    stream = _LiteLLMStream([chunk])
    underlying = stream.completion_stream
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=None, content=None)
            )
        ]
    )
    monkeypatch.setattr(bridge.litellm, "completion", lambda **_: stream)
    monkeypatch.setattr(
        bridge.litellm,
        "stream_chunk_builder",
        lambda *_args, **_kwargs: response,
    )

    assert _bare_model()._query([{"role": "user", "content": "x"}]) is response
    assert underlying.close_calls == 1
    assert stream.completion_stream is None


def test_query_closes_underlying_sync_stream_on_iteration_error(
    monkeypatch,
) -> None:
    class BrokenStream(_LiteLLMStream):
        def __iter__(self):
            raise RuntimeError("stream failed")

    stream = BrokenStream([])
    underlying = stream.completion_stream
    monkeypatch.setattr(bridge.litellm, "completion", lambda **_: stream)

    with pytest.raises(RuntimeError, match="stream failed"):
        _bare_model()._query([{"role": "user", "content": "x"}])

    assert underlying.close_calls == 1
    assert stream.completion_stream is None


@pytest.mark.parametrize(
    ("content", "expected_command"),
    [
        (
            '```json\n{"name":"bash","arguments":{"command":"git diff"}}\n```',
            "git diff",
        ),
        (
            '<response>\n```bash\n{"name":"bash","arguments":'
            '{"command":"rg SECRET_KEY django"}}\n```\n</response>',
            "rg SECRET_KEY django",
        ),
        (
            "I will inspect first.\n```bash\nsed -n '1,80p' django/core/signing.py\n```",
            "sed -n '1,80p' django/core/signing.py",
        ),
        (
            "First enter the repo.\n```bash\ncd /testbed\n```\nThen inspect."
            "\n```bash\nrg SECRET_KEY_FALLBACKS django\n```",
            "rg SECRET_KEY_FALLBACKS django",
        ),
        (
            '<tool_call>\n{"name":"bash","arguments":'
            '{"command":"python -m compileall django"}}\n</tool_call>',
            "python -m compileall django",
        ),
        (
            r'''{"name": "bash", "arguments": {"command": "find . '''
            r'''-exec sed -i 's/x/os.environ.get(\'SECRET_KEY\')/' {} \;"}}''',
            "find . -exec sed -i 's/x/os.environ.get('SECRET_KEY')/' {} \\;",
        ),
        (
            r'''```bash
{"name":"bash","arguments":{"command":"find . -exec sed -i '''
            r''''s/x/os.environ.get('SECRET_KEY')/' {} \;"}}
```''',
            "find . -exec sed -i 's/x/os.environ.get('SECRET_KEY')/' {} \\;",
        ),
    ],
)
def test_attach_embedded_tool_call_accepts_observed_qwen_formats(
    content: str, expected_command: str
) -> None:
    message = SimpleNamespace(tool_calls=None, content=content)

    bridge.BridgeReuseLitellmModel._attach_embedded_tool_call(
        message, "call_test"
    )

    assert len(message.tool_calls) == 1
    call = message.tool_calls[0]
    assert call.function.name == "bash"
    assert json.loads(call.function.arguments) == {"command": expected_command}


def test_attach_embedded_tool_call_does_not_invent_action_from_prose() -> None:
    message = SimpleNamespace(
        tool_calls=None,
        content="I should inspect the repository before changing anything.",
    )

    bridge.BridgeReuseLitellmModel._attach_embedded_tool_call(
        message, "call_test"
    )

    assert message.tool_calls is None


def test_attach_embedded_tool_call_can_break_format_loop_with_readonly_notice() -> None:
    content = "Inspect the relevant section of the code before editing."
    message = SimpleNamespace(tool_calls=None, content=content)

    bridge.BridgeReuseLitellmModel._attach_embedded_tool_call(
        message,
        "call_test",
        recover_unparsed_output_with_notice=True,
    )

    assert len(message.tool_calls) == 1
    call = message.tool_calls[0]
    assert call.function.name == "bash"
    arguments = json.loads(call.function.arguments)
    assert arguments["command"].startswith("printf")
    assert "changed nothing" in arguments["command"]
    assert message.content == content


def test_attach_embedded_tool_call_replaces_standalone_testbed_cd() -> None:
    message = SimpleNamespace(
        tool_calls=None,
        content='{"name":"bash","arguments":{"command":"cd /testbed"}}',
    )

    bridge.BridgeReuseLitellmModel._attach_embedded_tool_call(
        message, "call_test"
    )

    arguments = json.loads(message.tool_calls[0].function.arguments)
    assert arguments["command"].startswith("pwd;")
    assert "already starts in /testbed" in arguments["command"]


@pytest.mark.parametrize(
    "command",
    [
        "apt-get update && apt-get install ripgrep",
        "sudo apt install -y ripgrep",
        "pip install ripgrep",
    ],
)
def test_attach_embedded_tool_call_replaces_offline_package_install(
    command: str,
) -> None:
    message = SimpleNamespace(
        tool_calls=None,
        content=json.dumps(
            {"name": "bash", "arguments": {"command": command}}
        ),
    )

    bridge.BridgeReuseLitellmModel._attach_embedded_tool_call(
        message, "call_test"
    )

    arguments = json.loads(message.tool_calls[0].function.arguments)
    assert arguments["command"].startswith("printf")
    assert "container is offline" in arguments["command"]
    assert "Use existing grep/find/sed" in arguments["command"]
