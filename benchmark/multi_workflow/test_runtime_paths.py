from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.multi_workflow.runtime_paths import RuntimePaths


def test_runtime_paths_follow_environment(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "CodeMAS_Project"
    runtime = tmp_path / "impactkv-runtime"
    model = tmp_path / "models" / "model"
    for name in (
        "IMPACTKV_ARTIFACTS",
        "IMPACTKV_REPORTS",
        "IMPACTKV_POPULATION",
        "IMPACTKV_ENROOT_IMAGE_DIR",
        "IMPACTKV_ENROOT_IMAGE_INDEX",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IMPACTKV_HOME", str(home))
    monkeypatch.setenv("IMPACTKV_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("IMPACTKV_MODEL", str(model))
    paths = RuntimePaths.from_project(tmp_path / "checkout")
    assert paths.artifacts == home / "kvflow-artifacts"
    assert paths.runtime == runtime
    assert paths.model == model


def test_home_scope_rejects_external_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IMPACTKV_RUNTIME_ROOT", "/tmp/impactkv")
    paths = RuntimePaths.from_project(tmp_path / "checkout")
    with pytest.raises(ValueError, match="must be under"):
        paths.require_home_scoped_runtime()
