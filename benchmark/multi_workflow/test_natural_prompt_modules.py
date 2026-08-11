from __future__ import annotations

from benchmark.multi_workflow import motivate_v50_coding_provenance as m50
from benchmark.multi_workflow.natural_prompt_modules import (
    build_module_relations,
    classify_tool_result,
    render_natural_prompt_modules,
)


class CharacterTokenizer:
    """Tiny offset-capable tokenizer used to test exact partition behavior."""

    def encode(self, value: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(char) for char in value]

    def __call__(
        self,
        value: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        result: dict[str, object] = {"input_ids": self.encode(value, add_special_tokens)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(value))]
        return result


def _assistant(command: str, content: str = "I will inspect `src/parser.py`.") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"function": {"name": "computer", "arguments": {"command": command}}}
        ],
    }


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content}


def test_render_preserves_exact_prompt_and_separates_interpretation() -> None:
    tokenizer = CharacterTokenizer()
    base = [
        {"role": "system", "content": "Fix the repository."},
        {"role": "user", "content": "Parser crashes."},
    ]
    group = [
        _assistant("sed -n '1,200p' src/parser.py"),
        _tool("def parse(value):\n    return value\n" + "x" * 420 + "<returncode>0</returncode>"),
    ]
    ids, modules, _ = render_natural_prompt_modules(tokenizer, base, [group])
    expected = "".join(m50._render_message_literal(message) for message in base + group)
    expected += "<|im_start|>assistant\n"
    assert ids == tokenizer.encode(expected)
    assert [module["token_start"] for module in modules] == [
        0,
        *[module["token_end"] for module in modules[:-1]],
    ]
    assert [module["module_type"] for module in modules] == [
        "system_instruction",
        "task_specification",
        "assistant_interpretation",
        "tool_command",
        "repository_code",
        "generation_marker",
    ]
    assert modules[2]["paths"] == ["src/parser.py"]
    assert modules[4]["paths"] == ["src/parser.py"]


def test_tool_classification_is_mechanical() -> None:
    read = [
        _assistant("cat src/parser.py", ""),
        _tool("x" * 420 + "<returncode>0</returncode>"),
    ]
    search = [
        _assistant("rg parse src/parser.py", ""),
        _tool("x" * 420 + "<returncode>0</returncode>"),
    ]
    test = [
        _assistant("pytest tests/test_parser.py", ""),
        _tool("1 passed<returncode>0</returncode>"),
    ]
    mutation = [
        _assistant("apply_patch <<'PATCH'\n*** Update File: src/parser.py\nPATCH", ""),
        _tool("Done!<returncode>0</returncode>"),
    ]
    assert classify_tool_result(read) == "repository_code"
    assert classify_tool_result(search) == "repository_search"
    assert classify_tool_result(test) == "test_or_execution_feedback"
    assert classify_tool_result(mutation) == "diff_or_mutation_feedback"


def test_short_successful_read_is_still_a_code_module() -> None:
    read = [
        _assistant("sed -n '1,20p' src/parser.py", ""),
        _tool("def parse():\n    pass\n<returncode>0</returncode>"),
    ]
    assert classify_tool_result(read) == "repository_code"


def test_explicit_multifile_headers_create_variable_natural_modules() -> None:
    tokenizer = CharacterTokenizer()
    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    content = (
        "==> src/a.py <==\ndef alpha():\n    pass\n"
        "==> src/b.py <==\ndef beta():\n    pass\n"
        + "x" * 420
        + "<returncode>0</returncode>"
    )
    group = [_assistant("cat src/a.py src/b.py", ""), _tool(content)]
    _, modules, _ = render_natural_prompt_modules(tokenizer, base, [group])
    code = [module for module in modules if module["module_type"] == "repository_code"]
    assert len(code) == 2
    assert code[0]["paths"] == ["src/a.py"]
    assert code[1]["paths"] == ["src/b.py"]
    assert code[0]["natural_length"] != code[1]["natural_length"]


def test_mutation_invalidates_prior_code_and_advances_epoch() -> None:
    tokenizer = CharacterTokenizer()
    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    read = [
        _assistant("cat src/parser.py", ""),
        _tool("def parse():\n    pass\n" + "x" * 420 + "<returncode>0</returncode>"),
    ]
    mutate = [
        _assistant("apply_patch <<'PATCH'\n*** Update File: src/parser.py\nPATCH", ""),
        _tool("Done!<returncode>0</returncode>"),
    ]
    followup = [_assistant("cat src/other.py", ""), _tool("short<returncode>0</returncode>")]
    _, modules, _ = render_natural_prompt_modules(tokenizer, base, [read, mutate, followup])
    code = next(module for module in modules if module["module_type"] == "repository_code")
    final_command = [module for module in modules if module["module_type"] == "tool_command"][-1]
    assert code["invalidating_event"]["group_index"] == 1
    assert final_command["repository_epoch"] == 1


def test_relations_include_paths_symbols_grounding_and_failure_sequence() -> None:
    modules = [
        {
            "module_id": "m0",
            "module_type": "repository_code",
            "paths": ["src/parser.py"],
            "symbols": ["parse"],
            "source_request_index": 1,
            "parent_interaction_id": "i1",
            "repository_epoch": 0,
        },
        {
            "module_id": "m1",
            "module_type": "test_or_execution_feedback",
            "paths": ["tests/test_parser.py"],
            "symbols": ["parse"],
            "source_request_index": 2,
            "parent_interaction_id": "i2",
            "repository_epoch": 0,
        },
        {
            "module_id": "m2",
            "module_type": "assistant_interpretation",
            "paths": ["src/parser.py"],
            "symbols": ["parse"],
            "source_request_index": 3,
            "parent_interaction_id": "i3",
            "repository_epoch": 0,
            "grounding_module_ids": ["m0"],
        },
    ]
    relations = build_module_relations(modules)
    code_to_interpretation = next(
        row for row in relations if row["key_module_id"] == "m0" and row["query_module_id"] == "m2"
    )
    failure_to_action = next(
        row for row in relations if row["key_module_id"] == "m1" and row["query_module_id"] == "m2"
    )
    assert code_to_interpretation["exact_path"]
    assert code_to_interpretation["shared_symbol"]
    assert code_to_interpretation["interpretation_grounding"]
    assert failure_to_action["failure_to_next_action"]
