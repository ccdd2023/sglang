from __future__ import annotations

import json

from benchmark.multi_workflow.selective_ast_reuse import (
    build_selective_policy,
    content_signature,
    extract_codebase_files,
    select_spans,
    split_python_file,
)


SAMPLE_CODE = """
import math

class Solver:
    def helper(self, x):
        if x < 0:
            return -x
        return x

def top_level(y):
    return y + 1
""".strip()


def test_extract_codebase_files_from_fenced_marker():
    prompt = f"## code_base: pkg/foo.py\n```python\n{SAMPLE_CODE}\n```\n"
    files = extract_codebase_files(prompt)
    assert len(files) == 1
    assert files[0].path == "pkg/foo.py"
    assert "class Solver" in files[0].text


def test_extract_codebase_files_from_json_shape():
    prompt = json.dumps({"code_base": [{"path": "pkg/foo.py", "content": SAMPLE_CODE}]})
    files = extract_codebase_files(prompt)
    assert len(files) == 1
    assert files[0].path == "pkg/foo.py"
    assert "def top_level" in files[0].text


def test_split_python_file_extracts_function_method_class_and_control_block():
    policy = build_selective_policy(
        {
            "function": {"p90": 0.41},
            "method": {"p90": 0.40},
            "class": {"p90": 0.56, "max": 0.77},
            "control_block": {"p90": 0.46},
            "file_prefix": {"p90": 0.46},
        }
    )
    spans = split_python_file("pkg/foo.py", SAMPLE_CODE, policy)
    granularities = {span.granularity for span in spans}
    assert {"function", "method", "class", "control_block", "file_prefix"} <= granularities
    selected = select_spans(spans, "selective_function_method_reuse")
    assert selected
    assert {span.granularity for span in selected} <= {"function", "method"}


def test_content_signature_rejects_literal_change():
    before = "def f():\n    return 1\n"
    after = "def f():\n    return 2\n"
    assert content_signature(before) != content_signature(after)


def test_whole_file_mode_selects_file_prefix_only():
    policy = build_selective_policy({"function": {"p90": 0.41}, "file_prefix": {"p90": 0.46}})
    spans = split_python_file("pkg/foo.py", SAMPLE_CODE, policy)
    selected = select_spans(spans, "whole_file_reuse_all")
    assert selected
    assert selected[0].granularity == "file_prefix"


def test_selective_spans_do_not_overlap_nested_functions():
    code = """
class Stream:
    def iter_chunks(self):
        def generate():
            return b"x"
        return generate()
""".strip()
    policy = build_selective_policy({"method": {"p90": 0.40}, "function": {"p90": 0.41}})
    spans = split_python_file("pkg/stream.py", code, policy)
    selected = select_spans(spans, "selective_function_method_reuse")
    assert len(selected) == 1
    assert selected[0].name.endswith(":iter_chunks:2-5")
