from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from benchmark.multi_workflow.audit_search_result_file_modules import search_sections


def test_search_sections_use_literal_file_boundaries() -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    group = [
        {
            "role": "tool",
            "content": (
                "<returncode>0</returncode><output>\n"
                "pkg/a.py:10:def a():\n"
                "pkg/a.py:11:    return 1\n"
                "pkg/b.py:20:def b():\n"
                "</output>"
            ),
        }
    ]
    rows = search_sections(group, tokenizer)
    assert [row["path"] for row in rows] == ["pkg/a.py", "pkg/b.py"]
    assert [row["lines"] for row in rows] == [2, 1]
