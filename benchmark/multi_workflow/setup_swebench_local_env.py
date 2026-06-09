#!/usr/bin/env python3
"""Set up and test selected SWE-bench instances without Docker.

This is a pragmatic fallback for machines where the Docker daemon exists but
the current user cannot access `/var/run/docker.sock`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT / "results" / "repo_level_datasets" / "swe_verified_3_instances.json"
DEFAULT_WORKDIR = PROJECT / "results" / "swebench_local_envs"
DEFAULT_CONDA = Path("/home/gfy/miniconda3/bin/conda")


def _clean_user_env(env: dict[str, str]) -> dict[str, str]:
    """Strip KVCOMM/.venv and VIRTUAL_ENV from the inherited environment.

    The user's shell sets VIRTUAL_ENV=/home/gfy/KVCOMM/.venv and has
    /home/gfy/KVCOMM/.venv/bin early in PATH. KVCOMM/.venv is a
    --system-site-packages venv created from /home/gfy/.conda/envs/sglang-kvflow,
    so the inherited python is the same Python 3.12 as the sglang-kvflow
    conda env but the site-packages are KVCOMM's (which contains the
    dev/RC pytest build pointing at
    results/swebench_local_envs/repos/pytest-dev__pytest-7490/src). When
    we run `conda run -n swe_X python ...` inside that shell, conda
    appends the env's bin to the *end* of PATH and python/pip still
    resolve to the KVCOMM shim first — so `pip install pytest` lands in
    KVCOMM (not in swe_X) and `pytest` then runs the broken dev build,
    which crashes under Python 3.12's stricter AST handling with
    `TypeError: required field "lineno" missing from alias` in
    _pytest/assertion/rewrite.py. Strip both before any subprocess.run.
    """
    out = dict(env)
    out.pop("VIRTUAL_ENV", None)
    out.pop("PYTHONHOME", None)
    new_path_parts = [
        p for p in out.get("PATH", "").split(":")
        if p and "/KVCOMM/.venv" not in p
    ]
    out["PATH"] = ":".join(new_path_parts)
    return out


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int | None = None):
    print(f"$ {' '.join(shlex.quote(x) for x in cmd)}")
    base_env = env if env is not None else os.environ.copy()
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=_clean_user_env(base_env), text=True, timeout=timeout)


def run_checked(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int | None = None):
    result = run(cmd, cwd=cwd, env=env, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


def load_instance(dataset_path: Path, instance_id: str) -> dict[str, Any]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    for row in data:
        if row["instance_id"] == instance_id:
            return row
    raise KeyError(f"instance not found: {instance_id}")


def env_exists(conda: Path, env_name: str) -> bool:
    result = subprocess.run([str(conda), "env", "list"], text=True, capture_output=True)
    return any(line.split() and line.split()[0] == env_name for line in result.stdout.splitlines())


def ensure_repo(instance: dict[str, Any], repo_dir: Path):
    repo = instance["repo"]
    url = f"https://github.com/{repo}.git"
    if not (repo_dir / ".git").exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run_checked(["git", "clone", url, str(repo_dir)])
    run_checked(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
    run_checked(["git", "checkout", "--force", instance["base_commit"]], cwd=repo_dir)
    run_checked(["git", "clean", "-fdx"], cwd=repo_dir)


def write_patch(path: Path, content: str):
    path.write_text(content or "", encoding="utf-8")


def apply_patch_file(repo_dir: Path, patch_path: Path):
    if patch_path.read_text(encoding="utf-8").strip():
        run_checked(["git", "apply", str(patch_path)], cwd=repo_dir)


def ensure_local_native_deps(conda: Path, env_name: str, repo: str, repo_dir: Path):
    if repo == "matplotlib/matplotlib":
        run_checked(
            [
                str(conda),
                "install",
                "-y",
                "-n",
                env_name,
                "freetype=2.10.4",
                "pkg-config",
                "qhull",
                "libpng",
            ],
            cwd=repo_dir,
        )


def normalize_pip_packages(repo: str, pip_packages: list[str]) -> list[str]:
    packages = list(pip_packages)
    if repo == "scikit-learn/scikit-learn":
        normalized = []
        saw_cython = False
        for package in packages:
            if package.lower() == "cython":
                normalized.append("Cython<3")
                saw_cython = True
            else:
                normalized.append(package)
        if not saw_cython:
            normalized.insert(0, "Cython<3")
        packages = normalized
    return packages


def ensure_test_runner(conda: Path, env_name: str, specs: dict[str, Any], repo_dir: Path):
    test_cmd = specs.get("test_cmd", "pytest -rA")
    if not shlex.split(test_cmd) or shlex.split(test_cmd)[0] != "pytest":
        return
    pytest_package = "pytest<7" if str(specs.get("python", "")).startswith("3.6") else "pytest"
    # Use run_in_env (not run_checked) so the conda env's python/pip is
    # actually used, not the KVCOMM/.venv shim that the user's shell PATH
    # exposes. See run_in_env docstring for the full reasoning.
    run_in_env(conda, env_name, ["python", "-m", "pip", "install", pytest_package], cwd=repo_dir)


def should_skip_pre_install(command: str) -> bool:
    tokens = shlex.split(command)
    if os.geteuid() == 0 or not tokens:
        return False
    root_only_markers = {"apt", "apt-get", "sudo", "locale-gen"}
    if any(token in root_only_markers for token in tokens):
        return True
    if any(path in command for path in ["/etc/", "/testbed/"]):
        return True
    # The matplotlib local fallback installs qhull through conda instead of
    # replaying Docker-oriented source-build shell snippets.
    return "QHULL_" in command or "qhull" in command.lower()


def build_test_command(
    instance: dict[str, Any],
    specs: dict[str, Any],
    include_pass_to_pass: bool,
    max_fail_tests: int,
    test_targets: list[str] | None = None,
) -> list[str]:
    raw = specs.get("test_cmd", "pytest -rA")
    tests = list(test_targets or [])
    if not tests:
        tests = json.loads(instance.get("FAIL_TO_PASS", "[]"))
        if max_fail_tests > 0:
            tests = tests[:max_fail_tests]
        if include_pass_to_pass:
            tests += json.loads(instance.get("PASS_TO_PASS", "[]"))
    repo = instance["repo"]
    if repo == "astropy/astropy":
        return shlex.split(raw) + tests
    if repo == "matplotlib/matplotlib":
        return shlex.split(raw) + tests
    if repo == "django/django":
        return shlex.split(raw) + [django_test_label(test) for test in tests]
    return shlex.split(raw) + tests


def django_test_label(label: str) -> str:
    match = re.fullmatch(r"(?P<method>[^ ]+) \((?P<case>[^)]+)\)", label)
    if match:
        return f"{match.group('case')}.{match.group('method')}"
    return label


def run_in_env(
    conda: Path,
    env_name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
):
    # PATH/VIRTUAL_ENV cleanup happens in run() via _clean_user_env, so the
    # conda env's python/pip/pytest actually resolve to the env, not the
    # KVCOMM/.venv shim the user's shell exports.
    return run([str(conda), "run", "-n", env_name] + command, cwd=cwd, env=env, timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--mode", choices=["base", "gold", "candidate"], default="gold")
    parser.add_argument("--candidate-patch", type=Path)
    parser.add_argument("--include-pass-to-pass", action="store_true")
    parser.add_argument("--max-fail-tests", type=int, default=0)
    parser.add_argument("--test-target", action="append", default=[])
    parser.add_argument("--skip-pre-install", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    instance = load_instance(args.dataset, args.instance_id)
    specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][str(instance["version"])]
    env_name = f"swe_{args.instance_id.replace('__', '_').replace('-', '_')}_{args.mode}"
    repo_dir = args.workdir / "repos" / args.instance_id
    patch_dir = args.workdir / "patches" / args.instance_id
    patch_dir.mkdir(parents=True, exist_ok=True)

    ensure_repo(instance, repo_dir)

    if not env_exists(args.conda, env_name):
        run_checked([str(args.conda), "create", "-y", "-n", env_name, f"python={specs['python']}"])

    # Recent defaults-channel pip releases can drop compatibility with older
    # Python versions used by SWE-bench repositories.
    run_checked(
        [
            str(args.conda),
            "install",
            "-y",
            "-n",
            env_name,
            "pip<25",
            "wheel",
        ],
        cwd=repo_dir,
    )
    ensure_local_native_deps(args.conda, env_name, instance["repo"], repo_dir)

    if not args.skip_pre_install:
        for command in specs.get("pre_install", []):
            if should_skip_pre_install(command):
                print(f"Skipping root-only pre_install on local fallback: {command}")
                continue
            run_checked([str(args.conda), "run", "-n", env_name, "bash", "-lc", command], cwd=repo_dir)

    pip_packages = normalize_pip_packages(instance["repo"], specs.get("pip_packages", []))
    if pip_packages:
        run_checked([str(args.conda), "run", "-n", env_name, "python", "-m", "pip", "install"] + pip_packages, cwd=repo_dir)
    ensure_test_runner(args.conda, env_name, specs, repo_dir)

    test_patch = patch_dir / "test.patch"
    gold_patch = patch_dir / "gold.patch"
    write_patch(test_patch, instance.get("test_patch", ""))
    write_patch(gold_patch, instance.get("patch", ""))
    apply_patch_file(repo_dir, test_patch)
    if args.mode == "gold":
        apply_patch_file(repo_dir, gold_patch)
    if args.mode == "candidate":
        if args.candidate_patch is None:
            raise ValueError("--candidate-patch is required for --mode candidate")
        apply_patch_file(repo_dir, args.candidate_patch)

    install_cmd = specs.get("install")
    if install_cmd:
        run_checked([str(args.conda), "run", "-n", env_name, "bash", "-lc", install_cmd], cwd=repo_dir, timeout=args.timeout)

    test_cmd = build_test_command(
        instance,
        specs,
        args.include_pass_to_pass,
        args.max_fail_tests,
        args.test_target,
    )
    env = os.environ.copy()
    for command in specs.get("eval_commands", []):
        if command.startswith("export "):
            key, value = command[len("export ") :].split("=", 1)
            env[key] = value.strip("\"'")
    result = run_in_env(args.conda, env_name, test_cmd, repo_dir, env=env, timeout=args.timeout)

    out_dir = args.workdir / "reports" / args.instance_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "instance_id": args.instance_id,
        "repo": instance["repo"],
        "version": instance["version"],
        "mode": args.mode,
        "env_name": env_name,
        "repo_dir": str(repo_dir),
        "skip_pre_install": args.skip_pre_install,
        "test_cmd": test_cmd,
        "returncode": result.returncode,
    }
    (out_dir / f"{args.mode}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
