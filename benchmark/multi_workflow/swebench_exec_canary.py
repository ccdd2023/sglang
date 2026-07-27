#!/usr/bin/env python3
"""Small, executable SWE-bench canary with explicit failure attribution.

This driver is intentionally narrower than the official Docker harness.  It
uses already-created per-instance conda environments, but clones and mutates a
fresh repository under the requested output directory.  A run is trustworthy
only when the unpatched base fails and the gold patch passes the same tests.

The model can be asked for a unified diff or a complete target function.  In
function mode, a deterministic AST adapter creates the diff.  Patch
extraction/application failures, test failures, and infrastructure failures
are reported separately.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS


DEFAULT_DATASET = Path(
    "/home/gfy/CodeMAS_Project/sglang-kvflow/results/"
    "swebench_local_envs/expanded_30_discriminative_instances.json"
)
DEFAULT_REFERENCE_ROOT = Path(
    "/home/gfy/CodeMAS_Project/sglang-kvflow/results/swebench_local_envs/repos"
)
DEFAULT_OUTPUT = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_exec_canary_20260724"
)
DEFAULT_CONDA = Path("/home/gfy/miniconda3/bin/conda")


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 600,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
    ):
        env.pop(key, None)
    env["PATH"] = ":".join(
        part
        for part in env.get("PATH", "").split(":")
        if part and "/KVCOMM/.venv" not in part
    )
    env["PYTHONNOUSERSITE"] = "1"
    return env


def load_instance(dataset: Path, instance_id: str) -> dict[str, Any]:
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    for row in rows:
        if row["instance_id"] == instance_id:
            return row
    raise KeyError(f"instance not found: {instance_id}")


def patch_paths(patch: str) -> list[str]:
    found: list[str] = []
    for line in patch.splitlines():
        match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
        if match and match.group(2) not in found:
            found.append(match.group(2))
    return found


def repo_text(reference_repo: Path, commit: str, path: str) -> str:
    proc = run(["git", "show", f"{commit}:{path}"], cwd=reference_repo)
    if proc.returncode != 0:
        raise RuntimeError(f"git show failed for {path}: {proc.stderr[-1000:]}")
    return proc.stdout


def build_prompt(
    instance: dict[str, Any],
    reference_repo: Path,
    *,
    include_hints: bool,
    edit_protocol: str,
    target_symbol: str,
) -> tuple[str, list[str]]:
    paths = patch_paths(str(instance.get("patch") or ""))
    if not paths:
        raise ValueError("gold patch does not identify a target file")
    file_blocks = []
    for path in paths:
        content = repo_text(reference_repo, instance["base_commit"], path)
        file_blocks.append(f"===== {path} =====\n{content}")
    hints = str(instance.get("hints_text") or "").strip() if include_hints else ""
    hint_block = f"\n\nMaintainer discussion:\n{hints}" if hints else ""
    if edit_protocol == "function":
        output_contract = (
            f"Return ONLY the complete corrected Python function "
            f"`{target_symbol}`, including its `def` line. Do not return a "
            "diff, class wrapper, tests, prose, or Markdown fences. A "
            "deterministic AST adapter will replace that function and create "
            "the git diff."
        )
    else:
        output_contract = (
            "Return ONLY a unified git diff beginning with `diff --git`. "
            "Use exact context from the supplied base-commit files so "
            "`git apply` succeeds."
        )
    prompt = (
        "Fix the following real repository issue. "
        f"{output_contract} Keep the patch minimal and do not edit tests. "
        "Trace the code that directly creates the incorrect behavior before "
        "editing its callers. Do not insert a call that is already present.\n\n"
        f"Repository: {instance['repo']}\n"
        f"Base commit: {instance['base_commit']}\n\n"
        f"Issue:\n{instance['problem_statement']}"
        f"{hint_block}\n\n"
        "Relevant base-commit files:\n"
        + "\n\n".join(file_blocks)
    )
    return prompt, paths


def post_chat(
    endpoint: str,
    model: str,
    prompt: str,
    edit_protocol: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    if edit_protocol == "function":
        output_contract = (
            "Output only the complete corrected target Python function; "
            "never output prose, a diff, a class wrapper, tests, or Markdown."
        )
    else:
        output_contract = (
            "Output only an applicable unified git diff; never output prose "
            "or Markdown."
        )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"You are a repository maintenance agent. {output_contract}",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[-2000:]}") from exc
    body["client_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return body


def response_text(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except Exception:
        return ""


def extract_diff(text: str) -> str:
    fenced = re.search(r"```(?:diff)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("diff --git ")
    if start < 0:
        return ""
    return text[start:].strip() + "\n"


def extract_function(text: str, symbol: str) -> str:
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    lines = textwrap.dedent(text).strip().splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(rf"^\s*(?:async\s+)?def\s+{re.escape(symbol)}\s*\(", line)
        ),
        None,
    )
    if start is None:
        return ""
    candidate = textwrap.dedent("\n".join(lines[start:])).strip() + "\n"
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return ""
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol
    ]
    if len(functions) != 1:
        return ""
    node = functions[0]
    return "\n".join(candidate.splitlines()[node.lineno - 1 : node.end_lineno]) + "\n"


def synthesize_function_patch(
    *,
    reference_repo: Path,
    output: Path,
    instance: dict[str, Any],
    target_path: str,
    target_symbol: str,
    function_text: str,
) -> tuple[str, str]:
    if not function_text:
        return "", "function_not_extracted"
    synthesis_repo = output / "repos" / "synthesis" / instance["instance_id"]
    fresh_repo(reference_repo, synthesis_repo, instance["base_commit"])
    source_path = synthesis_repo / target_path
    source = source_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return "", f"base_file_ast_parse_failed: {exc}"
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == target_symbol
    ]
    if len(matches) != 1:
        return "", f"target_symbol_match_count={len(matches)}"
    node = matches[0]
    source_lines = source.splitlines(keepends=True)
    original_def_line = source_lines[node.lineno - 1]
    indentation = original_def_line[: len(original_def_line) - len(original_def_line.lstrip())]
    replacement = textwrap.indent(textwrap.dedent(function_text).rstrip(), indentation) + "\n"
    source_lines[node.lineno - 1 : node.end_lineno] = [replacement]
    source_path.write_text("".join(source_lines), encoding="utf-8")
    proc = run(["git", "diff", "--no-ext-diff", "--", target_path], cwd=synthesis_repo)
    if proc.returncode != 0:
        return "", f"git_diff_failed: {proc.stderr[-1000:]}"
    if not proc.stdout.strip():
        return "", "empty_synthesized_diff"
    return proc.stdout.rstrip() + "\n", ""


def fresh_repo(reference_repo: Path, destination: Path, commit: str) -> None:
    if destination.exists():
        run(["rm", "-rf", str(destination)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    proc = run(
        ["git", "clone", "--no-hardlinks", "--quiet", str(reference_repo), str(destination)],
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"clone failed: {proc.stderr[-2000:]}")
    proc = run(["git", "checkout", "--quiet", "--force", commit], cwd=destination)
    if proc.returncode != 0:
        raise RuntimeError(f"checkout failed: {proc.stderr[-2000:]}")
    run(["git", "clean", "-fdx"], cwd=destination)


def write_runtime_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def apply_patch(repo: Path, patch_file: Path) -> tuple[bool, str]:
    proc = run(["git", "apply", "--check", str(patch_file)], cwd=repo)
    if proc.returncode != 0:
        return False, proc.stderr[-4000:]
    proc = run(["git", "apply", str(patch_file)], cwd=repo)
    return proc.returncode == 0, proc.stderr[-4000:]


def django_label(label: str) -> str:
    match = re.fullmatch(r"(?P<method>[^ ]+) \((?P<case>[^)]+)\)", label)
    return f"{match.group('case')}.{match.group('method')}" if match else label


def test_command(
    instance: dict[str, Any],
    pass_to_pass: int,
    max_fail_to_pass: int,
) -> list[str]:
    specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][str(instance["version"])]
    command = shlex.split(specs.get("test_cmd", "pytest -rA"))
    fail_to_pass = json.loads(instance.get("FAIL_TO_PASS", "[]"))
    if max_fail_to_pass > 0:
        fail_to_pass = fail_to_pass[:max_fail_to_pass]
    stable = sorted(json.loads(instance.get("PASS_TO_PASS", "[]")))[:pass_to_pass]
    labels = fail_to_pass + stable
    if instance["repo"] == "django/django":
        labels = [django_label(label) for label in labels]
    return command + labels


def evaluate(
    *,
    instance: dict[str, Any],
    reference_repo: Path,
    output: Path,
    conda: Path,
    mode: str,
    candidate_patch: Path | None,
    pass_to_pass: int,
    max_fail_to_pass: int,
    timeout: int,
) -> dict[str, Any]:
    instance_id = instance["instance_id"]
    repo = output / "repos" / mode / instance_id
    fresh_repo(reference_repo, repo, instance["base_commit"])
    patch_dir = output / "patches" / instance_id
    test_patch = patch_dir / "test.patch"
    gold_patch = patch_dir / "gold.patch"
    write_runtime_file(test_patch, str(instance.get("test_patch") or ""))
    write_runtime_file(gold_patch, str(instance.get("patch") or ""))

    applied_test, apply_error = apply_patch(repo, test_patch)
    if not applied_test:
        return {
            "mode": mode,
            "status": "INFRA_FAILURE",
            "reason": "test_patch_apply_failed",
            "apply_error": apply_error,
        }

    treatment_patch = None
    if mode == "gold":
        treatment_patch = gold_patch
    elif mode == "candidate":
        treatment_patch = candidate_patch
    if treatment_patch is not None:
        if not treatment_patch.exists() or not treatment_patch.read_text(encoding="utf-8").strip():
            return {
                "mode": mode,
                "status": "FORMAT_FAILURE",
                "reason": "empty_candidate_patch",
            }
        applied, apply_error = apply_patch(repo, treatment_patch)
        if not applied:
            return {
                "mode": mode,
                "status": "FORMAT_FAILURE" if mode == "candidate" else "INFRA_FAILURE",
                "reason": "treatment_patch_apply_failed",
                "apply_error": apply_error,
            }

    env_name = (
        f"swe_{instance_id.replace('__', '_').replace('-', '_')}_"
        f"{'candidate' if mode == 'candidate' else mode}"
    )
    command = test_command(instance, pass_to_pass, max_fail_to_pass)
    proc = run(
        [str(conda), "run", "-n", env_name] + command,
        cwd=repo,
        timeout=timeout,
        env=clean_env(),
    )
    return {
        "mode": mode,
        "status": "PASS" if proc.returncode == 0 else "TEST_FAILURE",
        "returncode": proc.returncode,
        "env_name": env_name,
        "test_command": command,
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--instance-id", default="psf__requests-1142")
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/home/gfy/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--test-timeout", type=int, default=600)
    parser.add_argument("--pass-to-pass", type=int, default=5)
    parser.add_argument(
        "--max-fail-to-pass",
        type=int,
        default=0,
        help="0 evaluates every FAIL_TO_PASS test.",
    )
    parser.add_argument(
        "--edit-protocol",
        choices=("diff", "function"),
        default="diff",
    )
    parser.add_argument("--target-symbol", default="prepare_content_length")
    parser.add_argument(
        "--include-hints",
        action="store_true",
        help="Include issue-thread hints; disabled by default to avoid noisy discussions.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only validate the base/gold oracle.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    instance = load_instance(args.dataset, args.instance_id)
    reference_repo = args.reference_root / args.instance_id
    if not (reference_repo / ".git").exists():
        raise FileNotFoundError(f"reference repository unavailable: {reference_repo}")

    prompt, target_paths = build_prompt(
        instance,
        reference_repo,
        include_hints=args.include_hints,
        edit_protocol=args.edit_protocol,
        target_symbol=args.target_symbol,
    )
    write_runtime_file(args.output / "prompt.txt", prompt)
    registration = {
        "dataset": str(args.dataset),
        "instance_id": args.instance_id,
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "target_paths": target_paths,
        "functional_metric": "all FAIL_TO_PASS plus frozen sorted PASS_TO_PASS prefix",
        "pass_to_pass_count": args.pass_to_pass,
        "max_fail_to_pass": args.max_fail_to_pass,
        "include_hints": args.include_hints,
        "edit_protocol": args.edit_protocol,
        "target_symbol": args.target_symbol,
        "no_prefetch": True,
        "old_dirty_checkout_modified": False,
    }
    write_runtime_file(
        args.output / "REGISTRATION.json",
        json.dumps(registration, indent=2, ensure_ascii=False) + "\n",
    )

    results: dict[str, Any] = {"registration": registration, "oracle": {}}
    for mode in ("base", "gold"):
        results["oracle"][mode] = evaluate(
            instance=instance,
            reference_repo=reference_repo,
            output=args.output,
            conda=args.conda,
            mode=mode,
            candidate_patch=None,
            pass_to_pass=args.pass_to_pass,
            max_fail_to_pass=args.max_fail_to_pass,
            timeout=args.test_timeout,
        )

    oracle_valid = (
        results["oracle"]["base"]["status"] == "TEST_FAILURE"
        and results["oracle"]["gold"]["status"] == "PASS"
    )
    results["oracle_valid"] = oracle_valid
    if not oracle_valid:
        results["status"] = "INVALID_ORACLE"
    elif args.skip_generation:
        results["status"] = "ORACLE_VALID"
    else:
        response = post_chat(
            args.endpoint,
            args.model,
            prompt,
            args.edit_protocol,
            args.max_tokens,
            args.request_timeout,
        )
        raw = response_text(response)
        adapter_error = ""
        if args.edit_protocol == "function":
            function_text = extract_function(raw, args.target_symbol)
            write_runtime_file(args.output / "candidate_function.py", function_text)
            candidate, adapter_error = synthesize_function_patch(
                reference_repo=reference_repo,
                output=args.output,
                instance=instance,
                target_path=target_paths[0],
                target_symbol=args.target_symbol,
                function_text=function_text,
            )
        else:
            candidate = extract_diff(raw)
        write_runtime_file(args.output / "model_output.txt", raw)
        candidate_path = args.output / "candidate.patch"
        write_runtime_file(candidate_path, candidate)
        usage = response.get("usage") or {}
        results["generation"] = {
            "client_elapsed_ms": response.get("client_elapsed_ms"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "diff_extracted": bool(candidate),
            "adapter_error": adapter_error,
        }
        results["candidate"] = evaluate(
            instance=instance,
            reference_repo=reference_repo,
            output=args.output,
            conda=args.conda,
            mode="candidate",
            candidate_patch=candidate_path,
            pass_to_pass=args.pass_to_pass,
            max_fail_to_pass=args.max_fail_to_pass,
            timeout=args.test_timeout,
        )
        results["status"] = (
            "CANDIDATE_PASS"
            if results["candidate"]["status"] == "PASS"
            else "CANDIDATE_FAIL"
        )

    write_runtime_file(
        args.output / "RESULT.json",
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if oracle_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
