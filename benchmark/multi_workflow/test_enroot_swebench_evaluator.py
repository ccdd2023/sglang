from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("swebench")

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION

from benchmark.multi_workflow.enroot_swebench_evaluator import (
    evaluate_instance,
    write_json,
)


class _RejectingEnvironment:
    cleaned = False

    def __init__(self, **_kwargs) -> None:
        type(self).cleaned = False

    def write_text(self, _target: str, _text: str) -> dict:
        return {"returncode": 0, "output": ""}

    def execute(self, _action: dict, **_kwargs) -> dict:
        return {
            "returncode": 1,
            "output": "error: custom_coord.py: No such file or directory",
            "exception_info": "",
        }

    def cleanup(self) -> None:
        type(self).cleaned = True


def test_write_json_is_machine_readable_when_stdout_has_other_text(
    tmp_path: Path,
) -> None:
    output = tmp_path / "OFFICIAL_RESULT.json"
    write_json(output, {"returncode": 0, "report": {"resolved_instances": 1}})
    assert output.read_text(encoding="utf-8").startswith("{\n")
    assert not output.with_suffix(".json.partial").exists()


def test_unapplicable_model_patch_is_unresolved_not_infrastructure_error(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "benchmark.multi_workflow.enroot_swebench_evaluator.make_test_spec",
        lambda *_args, **_kwargs: SimpleNamespace(
            instance_image_key="sweb.eval.x86_64.owner_1776_repo-1:latest",
            eval_script="exit 0\n",
        ),
    )
    monkeypatch.setattr(
        "benchmark.multi_workflow.enroot_swebench_evaluator.EnrootEnvironment",
        _RejectingEnvironment,
    )
    result = evaluate_instance(
        instance={KEY_INSTANCE_ID: "owner__repo-1", "base_commit": "abc"},
        prediction={
            KEY_INSTANCE_ID: "owner__repo-1",
            KEY_MODEL: "model",
            KEY_PREDICTION: (
                "diff --git a/custom_coord.py b/custom_coord.py\n"
                "--- a/custom_coord.py\n"
                "+++ b/custom_coord.py\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
        },
        output_dir=tmp_path,
        timeout=30,
    )

    assert result["completed"] is True
    assert result["resolved"] is False
    assert result["invalid_patch"] is True
    assert result["evaluation_status"] == "invalid_model_patch"
    assert "error" not in result
    assert _RejectingEnvironment.cleaned is True
