from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import requests

from benchmark.multi_workflow.enroot_environment import (
    EnrootEnvironment,
    EnrootEnvironmentConfig,
    resolve_enroot_image,
)
from benchmark.multi_workflow.prepare_enroot_images import (
    docker_image_name,
    enroot_uri,
    optional_registry_digest,
    safe_image_filename,
)


def test_enroot_uri_uses_registry_separator() -> None:
    assert (
        enroot_uri("docker.io/swebench/example:latest")
        == "docker://registry-1.docker.io#swebench/example:latest"
    )


def test_image_name_matches_swebench_convention() -> None:
    row = {"instance_id": "owner__repo-123"}
    assert docker_image_name(row) == (
        "docker.io/swebench/sweb.eval.x86_64.owner_1776_repo-123:latest"
    )
    assert safe_image_filename(docker_image_name(row)).endswith(".sqsh")


def test_registry_digest_is_optional(monkeypatch) -> None:
    def fail(_reference: str) -> str:
        raise requests.ConnectionError("registry unavailable")

    monkeypatch.setattr(
        "benchmark.multi_workflow.prepare_enroot_images.registry_digest", fail
    )
    digest, error = optional_registry_digest("docker.io/swebench/task:latest")
    assert digest is None
    assert error == "ConnectionError: registry unavailable"


def test_nfs_whiteout_compatibility(tmp_path: Path) -> None:
    newest = tmp_path / "1"
    middle = tmp_path / "2"
    oldest = tmp_path / "3"
    for layer in (newest, middle, oldest):
        (layer / "etc" / "X11").mkdir(parents=True)
    (newest / "etc" / "X11" / ".wh..wh..opq").write_text("")
    (newest / "etc" / "X11" / "current").write_text("current")
    (middle / "etc" / "X11" / "middle").write_text("middle")
    (oldest / "etc" / "X11" / "old").write_text("old")
    (oldest / "etc" / "obsolete").write_text("old")
    (newest / "etc" / ".wh.obsolete").write_text("")

    helper = (
        Path(__file__).parent
        / "enroot_compat_bin"
        / "enroot-aufs2ovlfs"
    )
    subprocess.run([str(helper), str(newest)], check=True)

    assert (newest / "etc" / "X11" / "current").read_text() == "current"
    assert not (newest / "etc" / "X11" / ".wh..wh..opq").exists()
    assert not (newest / "etc" / ".wh.obsolete").exists()
    assert not any((middle / "etc" / "X11").iterdir())
    assert not any((oldest / "etc" / "X11").iterdir())
    assert not (oldest / "etc" / "obsolete").exists()


def test_resolve_image_from_index(tmp_path: Path) -> None:
    image = tmp_path / "task.sqsh"
    image.write_bytes(b"sqsh")
    index = tmp_path / "IMAGE_INDEX.json"
    index.write_text(
        json.dumps(
            {
                "images": {
                    "docker.io/swebench/task:latest": {
                        "sqsh_path": str(image)
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert resolve_enroot_image(
        "docker.io/swebench/task:latest", index
    ) == image


def test_execute_enters_namespace_and_preserves_cwd(monkeypatch, tmp_path: Path) -> None:
    environment = EnrootEnvironment.__new__(EnrootEnvironment)
    environment.config = EnrootEnvironmentConfig(
        image=str(tmp_path / "task.sqsh"),
        cwd="/testbed",
        env={"PAGER": "cat"},
        interpreter=["bash", "-lc"],
    )
    environment.process = SimpleNamespace(pid=123, poll=lambda: None)
    environment.runtime_path = tmp_path / "runtime"
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "ok\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = environment.execute({"command": "git status --short"})
    environment.process = None
    assert result["returncode"] == 0
    assert seen["command"][:4] == ["enroot", "exec", "123", "env"]
    assert "PATH=/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:" \
        "/usr/sbin:/usr/bin:/sbin:/bin" in seen["command"]
    assert "PAGER=cat" in seen["command"]
    assert seen["command"][-1] == "cd -- /testbed && git status --short"


def test_write_text_streams_into_namespace(monkeypatch, tmp_path: Path) -> None:
    environment = EnrootEnvironment.__new__(EnrootEnvironment)
    environment.config = EnrootEnvironmentConfig(
        image=str(tmp_path / "task.sqsh"),
        env={"PAGER": "cat"},
        interpreter=["bash", "-lc"],
    )
    environment.process = SimpleNamespace(pid=456, poll=lambda: None)
    environment.runtime_path = tmp_path / "runtime"
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = environment.write_text("/tmp/patch file.diff", "patch contents\n")
    environment.process = None

    assert result["returncode"] == 0
    assert seen["command"][:4] == ["enroot", "exec", "456", "env"]
    assert "PATH=/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:" \
        "/usr/sbin:/usr/bin:/sbin:/bin" in seen["command"]
    assert seen["command"][-1] == "umask 077 && cat > '/tmp/patch file.diff'"
    assert seen["kwargs"]["input"] == "patch contents\n"
