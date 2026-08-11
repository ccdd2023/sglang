"""mini-SWE-agent environment backed by a long-lived Enroot namespace."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from benchmark.multi_workflow.runtime_paths import RuntimePaths

try:
    from minisweagent.exceptions import Submitted
    from minisweagent.utils.serialize import recursive_merge
except ImportError:  # SWE-bench evaluation intentionally uses a separate venv
    class Submitted(RuntimeError):
        """Fallback used only by evaluator processes without mini-SWE-agent."""

    def recursive_merge(*values: dict[str, Any]) -> dict[str, Any]:
        """Minimal merge fallback; the agent venv uses mini-SWE-agent's helper."""

        merged: dict[str, Any] = {}
        for value in values:
            for key, item in value.items():
                if isinstance(item, dict) and isinstance(merged.get(key), dict):
                    merged[key] = recursive_merge(merged[key], item)
                else:
                    merged[key] = item
        return merged


class EnrootEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = Field(default_factory=dict)
    forward_env: list[str] = Field(default_factory=list)
    timeout: int = 30
    executable: str = Field(
        default_factory=lambda: os.getenv("MSWEA_ENROOT_EXECUTABLE", "enroot")
    )
    container_timeout: str = "2h"
    interpreter: list[str] = Field(default_factory=lambda: ["bash", "-lc"])
    startup_timeout: int = 30


def resolve_enroot_image(image: str, index_path: Path | None = None) -> Path:
    direct = Path(image).expanduser()
    if direct.is_file():
        return direct.resolve()
    paths = RuntimePaths.from_project(Path(__file__).resolve().parents[2])
    index_path = index_path or paths.enroot_image_index
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Enroot image index is missing: {index_path}; import {image!r} first"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    record = index.get("images", {}).get(image)
    if not record:
        raise KeyError(f"image {image!r} is absent from {index_path}")
    resolved = Path(record["sqsh_path"]).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"indexed Enroot image is missing: {resolved}")
    return resolved


class EnrootEnvironment:
    def __init__(
        self,
        *,
        config_class: type = EnrootEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.image_path = resolve_enroot_image(self.config.image)
        self.process: subprocess.Popen[str] | None = None
        self.runtime_path: Path | None = None
        self._start_container()

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(), platform.uname()._asdict(), kwargs
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    ),
                    "resolved_image": str(self.image_path),
                }
            }
        }

    def _host_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        project = Path(__file__).resolve().parents[2]
        paths = RuntimePaths.from_project(project)
        paths.require_home_scoped_runtime()
        job_id = os.getenv("SLURM_JOB_ID", "interactive")
        unique = f"{job_id}-{uuid.uuid4().hex[:12]}"
        runtime_base = Path(
            os.getenv(
                "IMPACTKV_ENROOT_RUNTIME_BASE",
                str(paths.runtime / "enroot" / "run"),
            )
        ).expanduser().resolve()
        user_home = Path.home().resolve()
        if not runtime_base.is_relative_to(user_home):
            raise ValueError(
                f"IMPACTKV_ENROOT_RUNTIME_BASE must be under {user_home}: "
                f"{runtime_base}"
            )
        self.runtime_path = runtime_base / unique
        self.runtime_path.mkdir(parents=True, mode=0o700)
        temp_path = Path(
            os.getenv(
                "ENROOT_TEMP_PATH", str(paths.runtime / "enroot" / "tmp")
            )
        ).expanduser().resolve()
        if not temp_path.is_relative_to(user_home):
            raise ValueError(
                f"ENROOT_TEMP_PATH must be under {user_home}: {temp_path}"
            )
        temp_path.mkdir(parents=True, exist_ok=True)
        env.update(
            {
                "ENROOT_RUNTIME_PATH": str(self.runtime_path),
                "ENROOT_TEMP_PATH": str(temp_path),
                "TMPDIR": str(temp_path),
                "TMP": str(temp_path),
                "TEMP": str(temp_path),
            }
        )
        return env

    def _start_container(self) -> None:
        command = [
            self.config.executable,
            "start",
            "--root",
            "--rw",
        ]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                command.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            command.extend(["-e", f"{key}={value}"])
        command.extend(
            [str(self.image_path), "sleep", self.config.container_timeout]
        )
        self.logger.debug("Starting Enroot namespace: %s", command)
        self.process = subprocess.Popen(
            command,
            env=self._host_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(
                    f"Enroot namespace exited with {self.process.returncode}: {stderr}"
                )
            probe = subprocess.run(
                [self.config.executable, "exec", str(self.process.pid), "true"],
                env=os.environ | {
                    "ENROOT_RUNTIME_PATH": str(self.runtime_path)
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode == 0:
                return
            time.sleep(0.2)
        self.cleanup()
        raise TimeoutError("Enroot namespace did not become ready")

    def execute(
        self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Enroot namespace is not running")
        command = action.get("command", "")
        workdir = cwd or self.config.cwd
        shell_command = f"cd -- {shlex_quote(workdir)} && {command}"
        cmd = [
            self.config.executable,
            "exec",
            str(self.process.pid),
            "env",
        ]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.append(f"{key}={value}")
        for key, value in self.config.env.items():
            cmd.append(f"{key}={value}")
        cmd.extend([*self.config.interpreter, shell_command])
        try:
            result = subprocess.run(
                cmd,
                env=os.environ
                | {"ENROOT_RUNTIME_PATH": str(self.runtime_path)},
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as exc:
            raw_output = getattr(exc, "output", None)
            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode("utf-8", errors="replace")
            output = {
                "output": raw_output or "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                },
            }
        self._check_finished(output)
        return output

    @staticmethod
    def _check_finished(output: dict[str, Any]) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": submission,
                    },
                }
            )

    def cleanup(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self.runtime_path is not None:
            shutil.rmtree(self.runtime_path, ignore_errors=True)
            self.runtime_path = None

    def __del__(self) -> None:
        self.cleanup()


def shlex_quote(value: str) -> str:
    """Small local wrapper to make tests and command construction explicit."""

    import shlex

    return shlex.quote(value)
