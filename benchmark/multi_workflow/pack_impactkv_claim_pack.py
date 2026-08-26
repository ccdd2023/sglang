#!/usr/bin/env python3
"""Build the off-cluster claim pack (frozen JSON + PLANs, no GPU dumps).

Run on the machine that still has kvflow-artifacts. Output is a gzip tarball
committed next to this file so collaborators can unpack without cluster access.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = HERE / "offcluster"
PACK_NAME = "impactkv-claim-pack.tar.gz"

FILES = [
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/RESULT.json",
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/MOTIVATION.json",
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/SLICES.json",
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/PLAN.json",
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/dense.json",
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/reuse.json",
    "impactkv_swebench_prerotated_file_modules_20260818/RESULT.json",
    "impactkv_swebench_prerotated_file_modules_20260818/SLICES.json",
    "impactkv_swebench_prerotated_file_modules_20260818/PLAN.json",
    "impactkv_swebench_prerotated_file_modules_20260818/dense.json",
    "impactkv_swebench_prerotated_file_modules_20260818/reuse.json",
    "impactkv_swebench_prerotated_file_modules_20260818/REGISTRATION.json",
    "impactkv_swebench_prerotated_file_modules_20260818/ANALYSIS.json",
    "impactkv_swebench_7b_sota_copiers_20260824/RESULT.json",
    "impactkv_swebench_7b_sota_copiers_20260824/RESULT.coding.json",
    "impactkv_swebench_7b_sota_copiers_20260824/RESULT.kvcomm.json",
    "impactkv_swebench_7b_sota_copiers_20260824/RESULT.cacheblend.json",
    "impactkv_swebench_7b_sota_copiers_20260824/COPIER_MOTIVATION.json",
    "impactkv_swebench_7b_sota_copiers_20260824/PLAN.coding.json",
    "impactkv_swebench_7b_sota_copiers_20260824/PLAN.kvcomm.json",
    "impactkv_swebench_7b_sota_copiers_20260824/PLAN.cacheblend.json",
    "impactkv_swebench_7b_sota_copiers_20260824/dense.json",
    "impactkv_swebench_7b_sota_copiers_20260824/kvcomm.json",
    "impactkv_swebench_7b_sota_copiers_20260824/cacheblend.json",
    "impactkv_swebench_7b_prefix_on_20260825/RESULT.json",
    "impactkv_swebench_template_prefetch_nextisland_20260821/RESULT.json",
    "impactkv_global_block_attention_20260806/frozen26_r2/RESULT.json",
    "impactkv_attention_sparsity_20260806/frozen20/RESULT.json",
    "impactkv_common_prompt_attention_kv_mechanism_20260813/FOUR_ARM_RESULT.json",
    (
        "impactkv_natural_code_cost_agent_expanded24_20260808/online/"
        "coding_natural_code_cost/full_24/CLIENT_LEDGER.jsonl"
    ),
    (
        "impactkv_natural_code_cost_agent_expanded24_20260808/online/"
        "coding_natural_code_cost/full_24/TELEMETRY.json"
    ),
    (
        "impactkv_natural_code_cost_agent_expanded24_20260808/online/"
        "coding_natural_code_cost/full_24/DYNAMIC_MANIFEST.json"
    ),
    "impactkv_common_prompt_attention_kv_mechanism_20260813/OBSERVATIONS_SGLANG.jsonl",
    "impactkv_common_prompt_attention_kv_mechanism_20260813/OBSERVATIONS_CACHEBLEND.jsonl",
    "impactkv_common_prompt_attention_kv_mechanism_20260813/OBSERVATIONS_KVCOMM.jsonl",
]


def _cluster_artifacts() -> Path:
    import os

    return Path(
        os.environ.get("IMPACTKV_ARTIFACTS", "/home/gfy/CodeMAS_Project/kvflow-artifacts")
    )


def main() -> None:
    src = _cluster_artifacts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pack = OUT_DIR / PACK_NAME
    manifest: dict[str, object] = {
        "schema_version": 1,
        "pack": PACK_NAME,
        "do_not_overwrite_frozen_result": True,
        "headline_job": "137185",
        "files": [],
    }
    missing = [rel for rel in FILES if not (src / rel).is_file()]
    if missing:
        raise FileNotFoundError("missing frozen files:\n" + "\n".join(missing))
    with tarfile.open(pack, "w:gz") as tar:
        for rel in FILES:
            path = src / rel
            tar.add(path, arcname=rel)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["files"].append(
                {"path": rel, "sha256": digest, "bytes": path.stat().st_size}
            )
    manifest["pack_sha256"] = hashlib.sha256(pack.read_bytes()).hexdigest()
    manifest["pack_bytes"] = pack.stat().st_size
    (OUT_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {pack} ({pack.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {OUT_DIR / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
