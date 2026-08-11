"""Portable filesystem contract for the active ImpactKV runners.

Historical experiment scripts intentionally retain their original absolute
paths.  New and actively maintained runners should resolve locations through
this module so the same checkout can run locally and in a Slurm home directory.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


MODEL_NAME = "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _executable_from_env(name: str, default: Path) -> Path:
    """Return an absolute executable path without dereferencing venv links."""

    return Path(
        os.path.abspath(Path(os.environ.get(name, str(default))).expanduser())
    )


@dataclass(frozen=True)
class RuntimePaths:
    project: Path
    home: Path
    artifacts: Path
    reports: Path
    population: Path
    model: Path
    mini_python: Path
    mini_executable: Path
    eval_python: Path
    runtime: Path
    enroot_images: Path
    enroot_image_index: Path

    @classmethod
    def from_project(cls, project: Path) -> "RuntimePaths":
        project = project.resolve()
        user_home = Path.home().resolve()
        home = _path_from_env(
            "IMPACTKV_HOME", user_home / "CodeMAS_Project"
        )
        runtime = _path_from_env(
            "IMPACTKV_RUNTIME_ROOT", user_home / "impactkv-runtime"
        )
        mini_root = _path_from_env(
            "IMPACTKV_MINI_VENV",
            user_home / ".venvs" / "mini-swe-agent-v2.3.0",
        )
        default_eval_python = (
            user_home / ".conda" / "envs" / "sglang-kvflow" / "bin" / "python"
        )
        if not default_eval_python.exists():
            default_eval_python = (
                user_home
                / "miniconda3"
                / "envs"
                / "sglang-kvflow"
                / "bin"
                / "python"
            )
        if not default_eval_python.exists():
            default_eval_python = Path(sys.executable)
        images = _path_from_env(
            "IMPACTKV_ENROOT_IMAGE_DIR", runtime / "enroot" / "images"
        )
        default_population = home / "datasets" / "swe_verified_500_instances.json"
        legacy_population = (
            home
            / "sglang-kvflow"
            / "results"
            / "repo_level_datasets"
            / "swe_verified_500_instances.json"
        )
        if not default_population.exists() and legacy_population.exists():
            default_population = legacy_population
        return cls(
            project=project,
            home=home,
            artifacts=_path_from_env(
                "IMPACTKV_ARTIFACTS", home / "kvflow-artifacts"
            ),
            reports=_path_from_env(
                "IMPACTKV_REPORTS", home / "kvflow-reports"
            ),
            population=_path_from_env(
                "IMPACTKV_POPULATION", default_population
            ),
            model=_path_from_env(
                "IMPACTKV_MODEL", user_home / "models" / MODEL_NAME
            ),
            mini_python=_executable_from_env(
                "IMPACTKV_MINI_PYTHON", mini_root / "bin" / "python"
            ),
            mini_executable=_executable_from_env(
                "IMPACTKV_MINI", mini_root / "bin" / "mini-extra"
            ),
            eval_python=_executable_from_env(
                "IMPACTKV_EVAL_PYTHON", default_eval_python
            ),
            runtime=runtime,
            enroot_images=images,
            enroot_image_index=_path_from_env(
                "IMPACTKV_ENROOT_IMAGE_INDEX", images / "IMAGE_INDEX.json"
            ),
        )

    def require_home_scoped_runtime(self) -> None:
        """Reject host runtime paths outside home on the Slurm deployment."""

        user_home = Path.home().resolve()
        checked = {
            "IMPACTKV_RUNTIME_ROOT": self.runtime,
            "IMPACTKV_ENROOT_IMAGE_DIR": self.enroot_images,
        }
        for name, value in checked.items():
            if not value.is_relative_to(user_home):
                raise ValueError(f"{name} must be under {user_home}: {value}")
