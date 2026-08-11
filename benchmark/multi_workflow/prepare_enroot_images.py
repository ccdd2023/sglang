#!/usr/bin/env python3
"""Import a frozen SWE-bench cohort into a home-scoped Enroot image index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchmark.multi_workflow.runtime_paths import RuntimePaths


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def docker_image_name(instance: dict[str, Any]) -> str:
    explicit = instance.get("image_name") or instance.get("docker_image")
    if explicit:
        return str(explicit).removeprefix("docker://")
    compatible = instance["instance_id"].replace("__", "_1776_")
    return (
        f"docker.io/swebench/sweb.eval.x86_64.{compatible}:latest".lower()
    )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "test.jsonl"
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return value
    for key in ("instances", "data", "test"):
        if isinstance(value.get(key), list):
            return value[key]
    raise ValueError(f"unsupported dataset shape: {path}")


def split_reference(reference: str) -> tuple[str, str, str]:
    reference = reference.removeprefix("docker://")
    first, slash, rest = reference.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        registry, repository_tag = first, rest
    else:
        registry, repository_tag = "docker.io", reference
    if registry == "docker.io" and "/" not in repository_tag:
        repository_tag = f"library/{repository_tag}"
    if "@" in repository_tag:
        repository, tag = repository_tag.split("@", 1)
        return registry, repository, tag
    tail = repository_tag.rsplit("/", 1)[-1]
    if ":" in tail:
        repository, tag = repository_tag.rsplit(":", 1)
    else:
        repository, tag = repository_tag, "latest"
    return registry, repository, tag


def enroot_uri(reference: str) -> str:
    registry, repository, tag = split_reference(reference)
    if registry == "docker.io":
        registry = "registry-1.docker.io"
    return f"docker://{registry}#{repository}:{tag}"


def registry_digest(reference: str) -> str | None:
    registry, repository, tag = split_reference(reference)
    if tag.startswith("sha256:"):
        return tag
    registry_host = "registry-1.docker.io" if registry == "docker.io" else registry
    headers = {
        "Accept": ", ".join(
            [
                "application/vnd.oci.image.index.v1+json",
                "application/vnd.docker.distribution.manifest.list.v2+json",
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.v2+json",
            ]
        )
    }
    url = f"https://{registry_host}/v2/{repository}/manifests/{tag}"
    response = requests.head(url, headers=headers, timeout=30)
    if response.status_code == 401 and registry == "docker.io":
        token_response = requests.get(
            "https://auth.docker.io/token",
            params={
                "service": "registry.docker.io",
                "scope": f"repository:{repository}:pull",
            },
            timeout=30,
        )
        token_response.raise_for_status()
        headers["Authorization"] = f"Bearer {token_response.json()['token']}"
        response = requests.head(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.headers.get("Docker-Content-Digest")


def optional_registry_digest(reference: str) -> tuple[str | None, str | None]:
    """Return provenance when available without invalidating an imported image."""

    try:
        return registry_digest(reference), None
    except (requests.RequestException, KeyError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def safe_image_filename(reference: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", reference).strip("_")[-120:]
    suffix = hashlib.sha256(reference.encode()).hexdigest()[:12]
    return f"{readable}-{suffix}.sqsh"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def import_image(reference: str, paths: RuntimePaths, force: bool) -> dict[str, Any]:
    paths.require_home_scoped_runtime()
    paths.enroot_images.mkdir(parents=True, exist_ok=True)
    output = paths.enroot_images / safe_image_filename(reference)
    if output.exists() and not force:
        digest, digest_error = optional_registry_digest(reference)
        return {
            "reference": reference,
            "registry_digest": digest,
            "registry_digest_error": digest_error,
            "sqsh_path": str(output),
            "sqsh_sha256": sha256(output),
            "bytes": output.stat().st_size,
            "imported_at_utc": None,
            "reused_existing": True,
        }
    partial = output.with_suffix(".partial.sqsh")
    partial.unlink(missing_ok=True)
    subprocess.run(
        ["enroot", "import", "--output", str(partial), enroot_uri(reference)],
        check=True,
    )
    partial.replace(output)
    digest, digest_error = optional_registry_digest(reference)
    return {
        "reference": reference,
        "registry_digest": digest,
        "registry_digest_error": digest_error,
        "sqsh_path": str(output),
        "sqsh_sha256": sha256(output),
        "bytes": output.stat().st_size,
        "imported_at_utc": utc_now(),
        "reused_existing": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--instances", help="comma-separated instance IDs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    paths = RuntimePaths.from_project(project)
    selected = set(filter(None, (args.instances or "").split(",")))
    rows = load_dataset(args.dataset)
    if selected:
        rows = [row for row in rows if row["instance_id"] in selected]
        missing = selected - {row["instance_id"] for row in rows}
        if missing:
            raise ValueError(f"instances absent from dataset: {sorted(missing)}")

    index = {"version": 1, "updated_at_utc": utc_now(), "images": {}}
    if paths.enroot_image_index.is_file():
        index = json.loads(paths.enroot_image_index.read_text(encoding="utf-8"))
    for row in rows:
        reference = docker_image_name(row)
        record = import_image(reference, paths, args.force)
        record["instance_ids"] = sorted(
            set(record.get("instance_ids", [])) | {row["instance_id"]}
        )
        index.setdefault("images", {})[reference] = record
        index["updated_at_utc"] = utc_now()
        atomic_json(paths.enroot_image_index, index)
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
