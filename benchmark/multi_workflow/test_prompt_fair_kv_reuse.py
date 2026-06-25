from __future__ import annotations

import json
from types import SimpleNamespace

from benchmark.multi_workflow import bench_selective_wholefile_reuse as selective
from benchmark.multi_workflow import build_selective_wholefile_manifest as manifest_builder
from benchmark.multi_workflow import bench_swe_generated_patch_kvcomm as patch_harness


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(f"{message['role']}:\n{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\nassistant:\n"
        return text

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=text.split())


def test_selective_target_prompt_is_identical_across_kv_modes():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    segments = [selective.CodeSegment("pkg/demo.py", "def parse_url(url):\n    return url\n")]
    tokenizer = DummyTokenizer()

    hashes = {}
    for mode in [
        "lossless_full_prefill",
        "whole_file_reuse_all",
        "selective_function_method_reuse",
        "selective_extended_reuse",
        "graph_aware_lossy",
    ]:
        messages = selective.build_wholefile_messages(instance, segments, mode)
        hashes[mode] = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert len(set(hashes.values())) == 1


def test_patch_harness_target_prompt_is_identical_when_agent_step_is_fixed():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
        "test_patch": "diff --git a/tests/test_demo.py b/tests/test_demo.py\n",
    }
    segments = [patch_harness.CodeSegment("pkg/demo.py", "def parse_url(url):\n    return url\n")]
    tokenizer = DummyTokenizer()

    hashes = []
    for _mode in ["lossless", "lossy", "graph_aware_lossy"]:
        messages = patch_harness.build_messages(
            instance,
            segments,
            "implementation target",
            "json-edit",
            "agenttemplatekv",
        )
        hashes.append(patch_harness.prompt_telemetry(tokenizer, messages)["target_prompt_sha1"])

    assert len(set(hashes)) == 1


def test_graph_reuse_segments_map_back_to_prompt_segments():
    prompt_segments = [
        patch_harness.CodeSegment(
            "pkg/demo.py",
            "class Prepared:\n    def prepare_url(self, url):\n        return url\n",
        )
    ]
    graph_segments = [
        patch_harness.CodeSegment(
            "pkg/demo.py",
            "def prepare_url(self, url):\n        return url",
        )
    ]

    mapped = patch_harness.graph_reuse_segments_in_prompt(prompt_segments, graph_segments)

    assert mapped == prompt_segments


def test_graph_segment_name_metadata_roundtrip_and_path_fallback():
    name = patch_harness.encode_graph_segment_name(
        "pkg/demo.py",
        "PreparedRequest.prepare_url",
        "call_neighborhood_1hop",
    )

    assert patch_harness.parse_graph_segment_name(name) == (
        "pkg/demo.py",
        "PreparedRequest.prepare_url",
        "call_neighborhood_1hop",
    )
    assert selective.parse_graph_segment_name(name) == (
        "pkg/demo.py",
        "PreparedRequest.prepare_url",
        "call_neighborhood_1hop",
    )

    prompt_segments = [
        patch_harness.CodeSegment(
            "pkg/demo.py",
            "class PreparedRequest:\n    def prepare_url(self, url):\n        return url\n",
        )
    ]
    graph_segments = [
        patch_harness.CodeSegment(
            name,
            "graph evidence text that is not directly inserted into the target prompt",
        )
    ]

    assert patch_harness.graph_reuse_segments_in_prompt(prompt_segments, graph_segments) == prompt_segments


def test_selective_graph_metadata_maps_target_symbol_to_prompt_ast_span():
    policy = {
        "include_file_prefix": True,
        "include_control_blocks": True,
        "include_function_method": True,
        "max_prefix_chars": 1000,
        "min_span_chars": 1,
    }
    whole = selective.CodeSegment(
        "pkg/demo.py",
        "class PreparedRequest:\n"
        "    def prepare_body(self, body):\n"
        "        return body\n\n"
        "    def prepare_url(self, url):\n"
        "        return url\n",
    )
    graph_name = selective.encode_graph_segment_name(
        "pkg/demo.py",
        "PreparedRequest.prepare_url",
        "call_neighborhood_1hop",
    )
    graph = selective.CodeSegment(
        graph_name,
        "graph-selected neighborhood; exact evidence stays out of the prompt",
    )

    mapped = selective.graph_mapped_reuse_spans([whole], policy, [graph])

    assert [span.name for span in mapped] == ["pkg/demo.py:method:prepare_url:5-6"]
    assert "prepare_body" not in mapped[0].text
    assert "prepare_url" in mapped[0].text


def test_selective_payload_prefetch_hints_align_with_anchor_token_spans():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    segment = selective.CodeSegment("pkg/demo.py", "def parse_url(url):\n    return url\n")
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [segment], "selective_function_method_reuse")
    args = SimpleNamespace(model="dummy-model", max_tokens=4)

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        [segment],
        "selective_function_method_reuse",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert payload["reuse_mode"] == "lossy"
    assert payload["next_agent_prefix"]
    assert payload["codebase_prefetch_hints"][0]["content_signature"] == segment.signature
    assert payload["code_anchor_token_spans"][0]["content_signature"] == segment.signature


def test_hybrid_calibration_rejects_anchor_selection():
    args = SimpleNamespace(
        _hybrid_calibration_policy_data={
            "cases": {
                "demo__case-1": {
                    "action": "reject",
                    "reason": "unit-test",
                }
            }
        }
    )
    segment = selective.CodeSegment("pkg/demo.py:bridge_prefix:file_start:1-3", "def parse_url(url):\n    return url\n")
    selected, selection = selective.apply_hybrid_calibration_to_selection(
        args,
        "demo__case-1",
        selective.HYBRID_CODE_AWARE_MODE,
        [segment],
        {
            "selected_span_count": 1,
            "selected_span_count_by_granularity": {"bridge_prefix": 1},
            "estimated_reused_tokens": 4,
            "decision_reason_counts": {"reuse:hybrid_large_function_bridge": 1},
        },
    )

    assert selected == []
    assert selection["hybrid_calibration_rejected"] is True
    assert selection["estimated_reused_tokens"] == 0


def test_hybrid_calibration_cap_preserves_prompt_hash_and_sets_span_cap():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    segment = selective.CodeSegment("pkg/demo.py", "def parse_url(url):\n    return url\n")
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [segment], selective.HYBRID_CODE_AWARE_MODE)
    args = SimpleNamespace(
        model="dummy-model",
        max_tokens=4,
        selective_anchor_max_start_token=0,
        selective_anchor_max_span_tokens=0,
        selective_anchor_min_span_tokens=0,
        anchor_min_total_tokens=0,
        anchor_max_total_tokens=0,
        anchor_max_total_policy="reject",
        anchor_lowspan_max_tokens=0,
        anchor_lowspan_suffix_copy_cap=0,
        anchor_smallspan_max_tokens=0,
        anchor_smallspan_suffix_copy_cap=0,
        anchor_midspan_min_tokens=0,
        anchor_midspan_max_tokens=0,
        anchor_midspan_suffix_copy_cap=0,
        graph_anchor_lowspan_max_tokens=0,
        graph_anchor_lowspan_suffix_copy_cap=0,
        graph_anchor_smallspan_max_tokens=0,
        graph_anchor_smallspan_suffix_copy_cap=0,
        graph_anchor_midspan_min_tokens=0,
        graph_anchor_midspan_max_tokens=0,
        graph_anchor_midspan_suffix_copy_cap=0,
        _hybrid_calibration_policy_data={
            "cases": {
                "demo__case-1": {
                    "action": "cap",
                    "max_suffix_copy_len": 3,
                    "reason": "unit-test",
                }
            }
        },
    )

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        [segment],
        selective.HYBRID_CODE_AWARE_MODE,
        "target:demo",
        "demo__case-1",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert payload["code_anchor_token_spans"][0]["max_suffix_copy_len"] == 3
    assert payload["_driver_anchor_telemetry"]["hybrid_calibration_action"] == "cap"


def test_hybrid_rule_calibration_matches_selection_without_case_entry():
    args = SimpleNamespace(
        _hybrid_calibration_policy_data={
            "rules": [
                {
                    "name": "bridge_method_window",
                    "action": "cap",
                    "max_suffix_copy_len": 7,
                    "reason": "unit-rule",
                    "match": {
                        "selected_span_count_by_granularity": {"bridge_prefix": 1, "method": 1},
                        "min_estimated_reused_tokens": 10,
                        "max_estimated_reused_tokens": 20,
                    },
                }
            ],
            "default_action": "reject",
        }
    )
    selected, selection = selective.apply_hybrid_calibration_to_selection(
        args,
        "unseen__case-1",
        selective.HYBRID_CODE_AWARE_MODE,
        [selective.CodeSegment("pkg/demo.py:bridge_prefix:file_start:1-10", "def f():\n    return 1")],
        {
            "selected_span_count": 2,
            "selected_span_count_by_granularity": {"bridge_prefix": 1, "method": 1},
            "estimated_reused_tokens": 15,
            "decision_reason_counts": {"reuse:hybrid_large_function_bridge": 1},
        },
    )

    assert selected
    assert selection["hybrid_calibration_action"] == "cap"
    assert selection["hybrid_calibration_rule_name"] == "bridge_method_window"
    assert selection["hybrid_calibration_max_suffix_copy_len"] == 7


def test_hybrid_rule_calibration_default_rejects_unmatched_selection():
    args = SimpleNamespace(
        _hybrid_calibration_policy_data={
            "rules": [
                {
                    "name": "method_only",
                    "action": "cap",
                    "max_suffix_copy_len": 5,
                    "match": {"selected_span_count_by_granularity": {"method": 1}},
                }
            ],
            "default_action": "reject",
            "default_reason": "unit-default",
        }
    )
    selected, selection = selective.apply_hybrid_calibration_to_selection(
        args,
        "unseen__case-2",
        selective.HYBRID_CODE_AWARE_MODE,
        [selective.CodeSegment("pkg/demo.py:bridge_prefix:file_start:1-10", "def f():\n    return 1")],
        {
            "selected_span_count": 1,
            "selected_span_count_by_granularity": {"bridge_prefix": 1},
            "estimated_reused_tokens": 15,
            "decision_reason_counts": {"reuse:hybrid_large_function_bridge": 1},
        },
    )

    assert selected == []
    assert selection["hybrid_calibration_action"] == "reject"
    assert selection["hybrid_calibration_rule_name"] == "__default__"
    assert selection["hybrid_calibration_rejected"] is True


def test_hybrid_rule_calibration_can_match_anchor_name_regex():
    args = SimpleNamespace(
        _hybrid_calibration_policy_data={
            "rules": [
                {
                    "name": "requests_models_bridge",
                    "action": "cap",
                    "max_suffix_copy_len": 9,
                    "match": {
                        "selected_span_count_by_granularity": {"bridge_prefix": 1},
                        "selected_anchor_name_any_regex": [r"requests/models\.py:bridge_prefix"],
                    },
                }
            ],
            "default_action": "reject",
        }
    )
    selected, selection = selective.apply_hybrid_calibration_to_selection(
        args,
        "unseen__case-3",
        selective.HYBRID_CODE_AWARE_MODE,
        [selective.CodeSegment("requests/models.py:bridge_prefix:file_start:1-10", "def f():\n    return 1")],
        {
            "selected_span_count": 1,
            "selected_span_count_by_granularity": {"bridge_prefix": 1},
            "selected_anchor_names": ["requests/models.py:bridge_prefix:file_start:1-10"],
            "estimated_reused_tokens": 15,
            "decision_reason_counts": {"reuse:hybrid_large_function_bridge": 1},
        },
    )

    assert selected
    assert selection["hybrid_calibration_rule_name"] == "requests_models_bridge"
    assert selection["hybrid_calibration_max_suffix_copy_len"] == 9


def test_hybrid_rule_calibration_cap_flows_from_selection_to_payload():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    segment = selective.CodeSegment("pkg/demo.py", "def parse_url(url):\n    return url\n")
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [segment], selective.HYBRID_CODE_AWARE_MODE)
    args = SimpleNamespace(
        model="dummy-model",
        max_tokens=4,
        selective_anchor_max_start_token=0,
        selective_anchor_max_span_tokens=0,
        selective_anchor_min_span_tokens=0,
        anchor_min_total_tokens=0,
        anchor_max_total_tokens=0,
        anchor_max_total_policy="reject",
        anchor_lowspan_max_tokens=0,
        anchor_lowspan_suffix_copy_cap=0,
        anchor_smallspan_max_tokens=0,
        anchor_smallspan_suffix_copy_cap=0,
        anchor_midspan_min_tokens=0,
        anchor_midspan_max_tokens=0,
        anchor_midspan_suffix_copy_cap=0,
        graph_anchor_lowspan_max_tokens=0,
        graph_anchor_lowspan_suffix_copy_cap=0,
        graph_anchor_smallspan_max_tokens=0,
        graph_anchor_smallspan_suffix_copy_cap=0,
        graph_anchor_midspan_min_tokens=0,
        graph_anchor_midspan_max_tokens=0,
        graph_anchor_midspan_suffix_copy_cap=0,
        _hybrid_calibration_policy_data={},
    )
    selection = {
        "hybrid_calibration_policy_applied": True,
        "hybrid_calibration_action": "cap",
        "hybrid_calibration_rule_name": "rule-from-selection",
        "hybrid_calibration_reason": "unit-selection",
        "hybrid_calibration_max_suffix_copy_len": 6,
    }

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        [segment],
        selective.HYBRID_CODE_AWARE_MODE,
        "target:demo",
        "demo__case-1",
        selection,
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert payload["code_anchor_token_spans"][0]["max_suffix_copy_len"] == 6
    assert payload["_driver_anchor_telemetry"]["hybrid_calibration_rule_name"] == "rule-from-selection"


def test_selected_segments_for_mode_records_anchor_names():
    policy = {
        "include_file_prefix": False,
        "include_control_blocks": False,
        "include_function_method": True,
        "max_prefix_chars": 1000,
        "min_span_chars": 1,
    }
    whole = selective.CodeSegment("pkg/demo.py", "def parse_url(url):\n    return url\n")
    selected, selection = selective.selected_segments_for_mode(
        [whole],
        policy,
        "selective_function_method_reuse",
        [],
        SimpleNamespace(exclude_anchor_granularities=""),
        DummyTokenizer(),
    )

    assert selected
    assert selection["selected_anchor_names"] == [selected[0].name]


def test_hybrid_bridge_source_extended_can_seed_file_prefix_bridge():
    policy = {
        "include_file_prefix": True,
        "include_control_blocks": False,
        "include_function_method": True,
        "max_prefix_chars": 200,
        "min_span_chars": 1,
    }
    text = (
        "import os\n"
        "import sys\n\n"
        "VALUE = 1\n\n"
        "def parse_url(url):\n"
        "    return url\n"
    )
    whole = selective.CodeSegment("pkg/demo.py", text)
    args = SimpleNamespace(
        hybrid_bridge_source="extended",
        hybrid_bridge_anchor_max_tokens=0,
        hybrid_min_bridge_tokens=0,
        hybrid_max_bridge_tokens=0,
        lossy_max_planned_suffix_copy_len=0,
        hybrid_bridge_max_count_per_file=0,
        hybrid_risk_large_bridge_min_tokens=0,
        hybrid_risk_max_large_bridge_count=0,
        hybrid_risk_max_graph_tokens_for_large_bridge=0,
        graph_anchor_token_budget=0,
        graph_anchor_max_span_tokens=0,
        exclude_anchor_granularities="",
    )

    selected, selection = selective.selected_segments_for_mode(
        [whole],
        policy,
        selective.HYBRID_CODE_AWARE_MODE,
        [],
        args,
        DummyTokenizer(),
    )

    assert selected
    assert selection["hybrid_bridge_source"] == "extended"
    assert selection["selected_span_count_by_granularity"] == {"bridge_prefix": 1}
    assert selected[0].name.startswith("pkg/demo.py:bridge_prefix:file_start:")


def test_anchor_feature_from_name_parses_bridge_prefix():
    feature = selective.anchor_feature_from_name("requests/models.py:bridge_prefix:file_start:1-20")

    assert feature["path"] == "requests/models.py"
    assert feature["granularity"] == "bridge_prefix"
    assert feature["symbol"] == "file_start"
    assert feature["line_range"] == "1-20"


def test_anchor_task_overlap_features_detect_path_mentions():
    instance = {
        "problem_statement": "See requests/models.py#L401; InvalidUrl should be raised.",
        "FAIL_TO_PASS": '["tests/test_requests.py::test_invalid_url"]',
    }
    features = selective.anchor_task_overlap_features(
        instance,
        ["requests/models.py:bridge_prefix:file_start:1-20"],
    )

    assert features[0]["path_mentioned"] is True
    assert features[0]["basename_mentioned"] is True
    assert features[0]["lexical_overlap"] >= 2


def test_hybrid_rule_calibration_can_require_anchor_task_overlap():
    args = SimpleNamespace(
        _hybrid_calibration_policy_data={
            "rules": [
                {
                    "name": "task_mentions_anchor_path",
                    "action": "cap",
                    "max_suffix_copy_len": 11,
                    "match": {
                        "selected_span_count_by_granularity": {"bridge_prefix": 1},
                        "require_anchor_path_mentioned": True,
                    },
                }
            ],
            "default_action": "reject",
        }
    )
    selected, selection = selective.apply_hybrid_calibration_to_selection(
        args,
        "unseen__case-4",
        selective.HYBRID_CODE_AWARE_MODE,
        [selective.CodeSegment("requests/models.py:bridge_prefix:file_start:1-10", "def f():\n    return 1")],
        {
            "selected_span_count": 1,
            "selected_span_count_by_granularity": {"bridge_prefix": 1},
            "selected_anchor_names": ["requests/models.py:bridge_prefix:file_start:1-10"],
            "estimated_reused_tokens": 15,
            "any_anchor_path_mentioned": True,
            "decision_reason_counts": {"reuse:hybrid_large_function_bridge": 1},
        },
    )

    assert selected
    assert selection["hybrid_calibration_rule_name"] == "task_mentions_anchor_path"
    assert selection["hybrid_calibration_max_suffix_copy_len"] == 11


def test_selection_feature_dry_run_writes_anchor_features(tmp_path):
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    case = {
        "instance": instance,
        "segments": [selective.CodeSegment("pkg/demo.py", "def parse_url(url):\n    return url\n")],
        "graph_segments": [],
    }
    args = SimpleNamespace(
        out_dir=tmp_path,
        dataset="dataset.json",
        manifest="manifest.json",
        target_modes="selective_function_method_reuse",
        enable_graph_aware_lossy=False,
        enable_hybrid_code_aware_lossy=False,
        hybrid_calibration_policy="",
        _hybrid_calibration_policy_data={},
        exclude_anchor_granularities="",
        selection_min_estimated_reused_tokens=0,
    )
    policy = {
        "include_file_prefix": False,
        "include_control_blocks": False,
        "include_function_method": True,
        "max_prefix_chars": 1000,
        "min_span_chars": 1,
    }

    selective.write_selection_feature_dry_run(args, DummyTokenizer(), [case], policy)
    data = json.loads((tmp_path / "selection_features.json").read_text())

    assert data["rows"][0]["status"] == "ok"
    assert data["rows"][0]["selected_anchor_features"][0]["path"] == "pkg/demo.py"
    assert "max_anchor_lexical_overlap" in data["rows"][0]
    assert (tmp_path / "selection_features.csv").exists()


def test_bridge_prefix_anchor_preserves_target_prompt_and_aligns_signatures():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    whole = selective.CodeSegment(
        "pkg/demo.py",
        "import os\n\n"
        "def helper():\n"
        "    return os.getcwd()\n\n"
        "def parse_url(url):\n"
        "    return url\n",
    )
    selected = selective.CodeSegment(
        "pkg/demo.py:function:parse_url:6-7",
        "def parse_url(url):\n    return url",
    )
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [whole], "graph_aware_lossy")
    args = SimpleNamespace(model="dummy-model", max_tokens=4)

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    bridged = selective.bridge_prefix_anchors([whole], [selected])
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        bridged,
        "graph_aware_lossy",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert bridged[0].name == "pkg/demo.py:bridge_prefix:file_start:1-7"
    assert bridged[0].text.startswith("import os")
    assert bridged[0].text.endswith("return url")
    hint_sigs = {hint["content_signature"] for hint in payload["codebase_prefetch_hints"]}
    span_sigs = {span["content_signature"] for span in payload["code_anchor_token_spans"]}
    assert bridged[0].signature in hint_sigs
    assert hint_sigs == span_sigs


def test_bounded_bridge_anchor_uses_prompt_resident_window_without_prompt_change():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    whole = selective.CodeSegment(
        "pkg/demo.py",
        "import os\n"
        "import sys\n"
        "VALUE = 1\n"
        "OTHER = 2\n\n"
        "def helper():\n"
        "    return VALUE\n\n"
        "def parse_url(url):\n"
        "    cleaned = url.strip()\n"
        "    return cleaned\n",
    )
    selected = selective.CodeSegment(
        "pkg/demo.py:function:parse_url:9-11",
        "def parse_url(url):\n    cleaned = url.strip()\n    return cleaned",
    )
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [whole], "graph_aware_lossy")
    args = SimpleNamespace(model="dummy-model", max_tokens=4)

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    bridged = selective.bridge_prefix_anchors([whole], [selected], max_tokens=8)
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        bridged,
        "graph_aware_lossy",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert bridged[0].name.startswith("pkg/demo.py:bridge_window:bounded:")
    assert not bridged[0].text.startswith("import os")
    assert bridged[0].text.endswith("return cleaned")
    assert bridged[0].text in selective.prompt_text_for_messages(tokenizer, messages)
    hint_sigs = {hint["content_signature"] for hint in payload["codebase_prefetch_hints"]}
    span_sigs = {span["content_signature"] for span in payload["code_anchor_token_spans"]}
    assert bridged[0].signature in hint_sigs
    assert hint_sigs == span_sigs


def test_hybrid_can_use_bounded_bridge_window_without_prompt_change():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    whole = selective.CodeSegment(
        "pkg/demo.py",
        "import os\n"
        "import sys\n"
        "A = 1\n"
        "B = 2\n\n"
        "def unrelated():\n"
        "    return A + B\n\n"
        "def parse_url(url):\n"
        "    cleaned = url.strip()\n"
        "    if not cleaned:\n"
        "        return 'missing'\n"
        "    return cleaned\n",
    )
    policy = {"granularities": {"function": {"decision": "reuse"}}}
    all_spans = selective.split_python_file(whole.name, whole.text, policy)
    args = SimpleNamespace(
        hybrid_bridge_anchor_max_tokens=8,
        hybrid_min_bridge_tokens=0,
        hybrid_max_bridge_tokens=0,
        lossy_max_planned_suffix_copy_len=0,
        hybrid_risk_large_bridge_min_tokens=0,
        hybrid_risk_max_large_bridge_count=0,
        hybrid_risk_max_graph_tokens_for_large_bridge=0,
    )
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [whole], "hybrid_code_aware_lossy")

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    selected, selection = selective.hybrid_code_aware_segments(
        [whole],
        all_spans,
        policy,
        graph_segments=None,
        args=args,
        tokenizer=tokenizer,
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert selection["hybrid_bridge_anchor_max_tokens"] == 8
    assert selected
    assert all(":bridge_window:bounded:" in segment.name for segment in selected)
    assert any("def parse_url" in segment.text for segment in selected)
    assert all(segment.text in selective.prompt_text_for_messages(tokenizer, messages) for segment in selected)
    assert all(not segment.text.startswith("import os") for segment in selected)


def test_hybrid_bounded_bridge_can_keep_deepest_window_per_file():
    whole = selective.CodeSegment(
        "pkg/demo.py",
        "import os\n\n"
        "def first():\n"
        "    return 1\n\n"
        "def second():\n"
        "    return 2\n\n"
        "def third():\n"
        "    return 3\n",
    )
    policy = {"granularities": {"function": {"decision": "reuse"}}}
    all_spans = selective.split_python_file(whole.name, whole.text, policy)
    args = SimpleNamespace(
        hybrid_bridge_anchor_max_tokens=6,
        hybrid_bridge_max_count_per_file=1,
        hybrid_min_bridge_tokens=0,
        hybrid_max_bridge_tokens=0,
        lossy_max_planned_suffix_copy_len=0,
        hybrid_risk_large_bridge_min_tokens=0,
        hybrid_risk_max_large_bridge_count=0,
        hybrid_risk_max_graph_tokens_for_large_bridge=0,
    )

    selected, selection = selective.hybrid_code_aware_segments(
        [whole],
        all_spans,
        policy,
        graph_segments=None,
        args=args,
        tokenizer=DummyTokenizer(),
    )

    assert len(selected) == 1
    assert "third" in selected[0].text
    assert selected[0].name.startswith("pkg/demo.py:bridge_window:bounded:")
    assert selection["hybrid_bridge_max_count_per_file"] == 1
    assert selection["hybrid_bridge_count_pruned"] == 2


def test_payload_drops_non_prompt_resident_anchor_instead_of_failing():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    whole = selective.CodeSegment("pkg/demo.py", "def parse_url(url):\n    return url\n")
    missing = selective.CodeSegment("pkg/demo.py:function:missing:1-2", "def missing():\n    return 2")
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [whole], "selective_extended_reuse")
    args = SimpleNamespace(model="dummy-model", max_tokens=4)

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        [missing],
        "selective_extended_reuse",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert payload["reuse_mode"] == "lossless"
    assert "codebase_prefetch_hints" not in payload
    assert "code_anchor_token_spans" not in payload


def test_payload_drops_overlapping_anchor_that_builder_cursor_cannot_relocate():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    text = (
        "def outer():\n"
        "    value = 1\n"
        "    def inner():\n"
        "        return value\n"
        "    return inner()\n"
    )
    whole = selective.CodeSegment("pkg/demo.py", text)
    outer = selective.CodeSegment("pkg/demo.py:function:outer:1-5", text.rstrip())
    inner = selective.CodeSegment(
        "pkg/demo.py:function:inner:3-4",
        "    def inner():\n        return value",
    )
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [whole], "selective_extended_reuse")
    args = SimpleNamespace(model="dummy-model", max_tokens=4)

    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        [outer, inner],
        "selective_extended_reuse",
        "target:demo",
    )

    assert payload["reuse_mode"] == "lossy"
    assert len(payload["code_anchor_token_spans"]) == 1
    assert payload["code_anchor_token_spans"][0]["segment_name"] == outer.name


def test_graph_anchor_budget_filters_long_spans_without_prompt_change():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(
        instance,
        [
            selective.CodeSegment(
                "pkg/demo.py",
                "def small():\n    return 1\n\n"
                "def medium():\n    value = 1\n    return value\n\n"
                "def huge():\n    "
                + " ".join(f"x{i}" for i in range(40))
                + "\n",
            )
        ],
        "graph_aware_lossy",
    )
    spans = [
        SimpleNamespace(
            name="pkg/demo.py:function:huge:8-9",
            path="pkg/demo.py",
            text=" ".join(f"x{i}" for i in range(40)),
            granularity="function",
            start_line=8,
            end_line=9,
        ),
        SimpleNamespace(
            name="pkg/demo.py:function:medium:4-6",
            path="pkg/demo.py",
            text="def medium():\n    value = 1\n    return value",
            granularity="function",
            start_line=4,
            end_line=6,
        ),
    ]
    selected, info = selective.apply_graph_anchor_budget(
        spans,
        SimpleNamespace(graph_anchor_max_span_tokens=10, graph_anchor_token_budget=20),
    )
    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    payload = selective.make_payload(
        SimpleNamespace(model="dummy-model", max_tokens=4),
        tokenizer,
        messages,
        [selective.CodeSegment(span.name, span.text) for span in selected],
        "graph_aware_lossy",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert info["graph_anchor_budget_applied"] is True
    assert info["graph_anchor_filtered_long_count"] == 1
    assert [span.name for span in selected] == ["pkg/demo.py:function:medium:4-6"]
    hint_sigs = {hint["content_signature"] for hint in payload["codebase_prefetch_hints"]}
    span_sigs = {span["content_signature"] for span in payload["code_anchor_token_spans"]}
    assert hint_sigs == span_sigs


def test_selective_anchor_max_span_tokens_filters_metadata_only():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    long_segment = selective.CodeSegment("pkg/demo.py:function:long:1-1", " ".join(f"x{i}" for i in range(20)))
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [long_segment], "selective_function_method_reuse")
    args = SimpleNamespace(model="dummy-model", max_tokens=4, selective_anchor_max_span_tokens=5)

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        [long_segment],
        "selective_function_method_reuse",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert payload["reuse_mode"] == "lossless"
    assert "codebase_prefetch_hints" not in payload
    assert "code_anchor_token_spans" not in payload


def test_anchor_min_total_tokens_filters_metadata_only():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    small_segment = selective.CodeSegment("pkg/demo.py:function:small:1-2", "def f():\n    return 1")
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [small_segment], "graph_aware_lossy")
    args = SimpleNamespace(model="dummy-model", max_tokens=4, anchor_min_total_tokens=10)

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        [small_segment],
        "graph_aware_lossy",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert payload["reuse_mode"] == "lossless"
    assert "codebase_prefetch_hints" not in payload
    assert "code_anchor_token_spans" not in payload


def test_anchor_min_span_tokens_filters_metadata_only():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    small_segment = selective.CodeSegment("pkg/demo.py:function:small:1-2", "def f():\n    return 1")
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [small_segment], "graph_aware_lossy")
    args = SimpleNamespace(model="dummy-model", max_tokens=4, selective_anchor_min_span_tokens=10)

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        [small_segment],
        "graph_aware_lossy",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert payload["reuse_mode"] == "lossless"
    assert "codebase_prefetch_hints" not in payload
    assert "code_anchor_token_spans" not in payload


def test_selection_level_gate_filters_before_bridge_without_prompt_change():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    whole = selective.CodeSegment(
        "pkg/demo.py",
        "import os\n\n"
        "def parse_url(url):\n"
        "    return url\n",
    )
    selected = selective.CodeSegment(
        "pkg/demo.py:function:parse_url:3-4",
        "def parse_url(url):\n    return url",
    )
    selection = {
        "estimated_reused_tokens": 4,
        "selected_span_count": 1,
    }
    args = SimpleNamespace(
        model="dummy-model",
        max_tokens=4,
        selection_min_estimated_reused_tokens=10,
        enable_bridge_prefix_anchors=True,
    )
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [whole], "selective_extended_reuse")

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    gated, gated_selection = selective.apply_selection_level_gate(
        [selected], selection, args, "selective_extended_reuse"
    )
    bridged = selective.maybe_bridge_prefix_anchors(args, [whole], gated, "selective_extended_reuse")
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        bridged,
        "selective_extended_reuse",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert gated == []
    assert bridged == []
    assert gated_selection["selection_gate_rejected"] is True
    assert gated_selection["selection_gate_reason"] == "estimated_reused_tokens_below_10"
    assert payload["reuse_mode"] == "lossless"


def test_graph_bridge_can_be_disabled_without_prompt_change():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    whole = selective.CodeSegment(
        "pkg/demo.py",
        "import os\n\n"
        "def parse_url(url):\n"
        "    return url\n",
    )
    selected = selective.CodeSegment(
        "pkg/demo.py:function:parse_url:3-4",
        "def parse_url(url):\n    return url",
    )
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [whole], "graph_aware_lossy")
    args = SimpleNamespace(
        model="dummy-model",
        max_tokens=4,
        enable_bridge_prefix_anchors=True,
        bridge_anchor_max_tokens=0,
        disable_graph_bridge_prefix_anchors=True,
    )

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    graph_selected = selective.maybe_bridge_prefix_anchors(args, [whole], [selected], "graph_aware_lossy")
    extended_selected = selective.maybe_bridge_prefix_anchors(
        args, [whole], [selected], "selective_extended_reuse"
    )
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        graph_selected,
        "graph_aware_lossy",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert graph_selected == [selected]
    assert extended_selected[0].name == "pkg/demo.py:bridge_prefix:file_start:1-4"
    assert payload["code_anchor_token_spans"][0]["segment_name"] == selected.name


def test_graph_midspan_copy_cap_is_metadata_only():
    instance = {
        "instance_id": "demo__case-1",
        "repo": "demo/repo",
        "problem_statement": "Fix the URL validation bug.",
        "FAIL_TO_PASS": '["tests/test_demo.py::test_url"]',
    }
    segment = selective.CodeSegment(
        "pkg/demo.py:function:parse_url:1-2",
        "def parse_url(url):\n    return url",
    )
    tokenizer = DummyTokenizer()
    messages = selective.build_wholefile_messages(instance, [segment], "graph_aware_lossy")
    args = SimpleNamespace(
        model="dummy-model",
        max_tokens=4,
        graph_anchor_midspan_min_tokens=3,
        graph_anchor_midspan_max_tokens=10,
        graph_anchor_midspan_suffix_copy_cap=2,
    )

    before_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]
    payload = selective.make_payload(
        args,
        tokenizer,
        messages,
        [segment],
        "graph_aware_lossy",
        "target:demo",
    )
    after_hash = selective.prompt_telemetry(tokenizer, messages)["prompt_sha1"]

    assert before_hash == after_hash
    assert payload["reuse_mode"] == "lossy"
    assert payload["code_anchor_token_spans"][0]["max_suffix_copy_len"] == 2


def test_accuracy_bucket_thresholds():
    assert selective.accuracy_bucket(1.0) == "strict-safe"
    assert selective.accuracy_bucket(0.99996) == "strict-safe"
    assert selective.accuracy_bucket(0.90) == "lossy-acceptable"
    assert selective.accuracy_bucket(0.8999) == "aggressive-diagnostic"
    assert selective.accuracy_bucket(0.9499, threshold=0.95) == "aggressive-diagnostic"
    assert selective.accuracy_bucket(None) == "unknown"


def test_reuse_policy_name_tracks_pareto_experiment_mode():
    base = dict(
        lossy_stage_recompute_gap=False,
        lossy_recompute_gap=False,
        enable_bridge_prefix_anchors=False,
        lossy_max_zero_gap=None,
    )
    assert selective.reuse_policy_name(
        SimpleNamespace(**{**base, "lossy_recompute_gap": True, "lossy_stage_recompute_gap": True}),
        "lossless_full_prefill",
    ) == "lossless_full_prefill"
    assert selective.reuse_policy_name(SimpleNamespace(**base), "graph_aware_lossy") == "graph_aware_lossy"
    assert selective.reuse_policy_name(
        SimpleNamespace(**{**base, "lossy_recompute_gap": True}),
        "selective_function_method_reuse",
    ) == "context_aligned_safe"
    assert selective.reuse_policy_name(
        SimpleNamespace(**{**base, "lossy_recompute_gap": True, "lossy_stage_recompute_gap": True}),
        "graph_aware_lossy",
    ) == "context_aligned_stage_diag"


def test_target_modes_filters_enabled_modes():
    args = SimpleNamespace(enable_graph_aware_lossy=True, target_modes="lossless_full_prefill,graph_aware_lossy")
    assert selective.active_modes(args) == ["lossless_full_prefill", "graph_aware_lossy"]


def test_task_aware_manifest_file_selection_prefers_mentioned_file(tmp_path):
    utils = tmp_path / "requests" / "utils.py"
    models = tmp_path / "requests" / "models.py"
    utils.parent.mkdir()
    utils.write_text("def prepend_scheme_if_needed(url):\n    return url\n", encoding="utf-8")
    models.write_text(
        "def many_spans_a():\n    return 1\n\n"
        "def many_spans_b():\n    return 2\n\n"
        "def many_spans_c():\n    return 3\n",
        encoding="utf-8",
    )
    sample = {
        "files": [
            {"path": "requests/models.py", "local_path": str(models)},
            {"path": "requests/utils.py", "local_path": str(utils)},
        ]
    }
    instance = {
        "problem_statement": "Proxy authentication bug",
        "FAIL_TO_PASS": '["tests/test_utils.py::test_prepend_scheme_if_needed"]',
    }
    policy = {
        "include_file_prefix": False,
        "include_control_blocks": False,
        "include_function_method": True,
        "max_prefix_chars": 1000,
        "min_span_chars": 1,
    }

    selected = manifest_builder.select_files(
        sample,
        instance,
        policy,
        max_file_chars=10000,
        files_per_case=1,
        selection_strategy="task_aware",
    )

    assert selected[0]["path"] == "requests/utils.py"
    assert selected[0]["task_file_score"] > 0


def test_load_cases_respects_manifest_file_max_chars(tmp_path):
    source = tmp_path / "pkg" / "demo.py"
    source.parent.mkdir()
    source.write_text("0123456789abcdef", encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    manifest = tmp_path / "manifest.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "instance_id": "demo__case-1",
                    "repo": "demo/repo",
                    "problem_statement": "Fix demo",
                    "FAIL_TO_PASS": "[]",
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "instance_id": "demo__case-1",
                        "repo": "demo/repo",
                        "files": [
                            {
                                "path": "pkg/demo.py",
                                "local_path": str(source),
                                "max_file_chars": 6,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        dataset=dataset,
        manifest=manifest,
        start_index=0,
        max_cases=1,
        all_cases=False,
        enable_graph_aware_lossy=False,
        code_graph_bundle_manifest=tmp_path / "missing.jsonl",
        graph_bundle_role="planner",
        graph_bundle_policy="target_call_neighborhood",
        max_graph_bundle_chars=0,
        graph_bundles_per_case=1,
        prefer_graph_target_files=False,
        file_start_index=0,
        files_per_case=1,
        prefer_selective_files=False,
        max_complete_file_chars=0,
        max_file_chars=100,
    )

    cases = selective.load_cases(args, policy={})

    assert cases[0]["segments"][0].text == "012345"
