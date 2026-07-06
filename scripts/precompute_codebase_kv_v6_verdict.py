#!/usr/bin/env python3
"""Direction A v6 — Verdict-aligned preamble (R22b).

Builds on v4 (Direction A v3) by replacing the "Inspect...implementation risk"
instruction with a verdict-aligned one, so the precomputed prefix covers the
exact instruction the model will see under --task-mode verdict.

This puts VERDICT: PASS/FAIL discipline into the precompute KV, anchoring the
model's attention to format-stable generation.

Usage:
    python scripts/precompute_codebase_kv_v6_verdict.py \
        --model-path Qwen/Qwen2.5-Coder-7B-Instruct \
        --working-set-manifest ... \
        --pandas-src ... \
        --run-tag pandas_5case_v6_verdict
"""
from __future__ import annotations

import argparse, hashlib, json, logging, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.precompute_codebase_kv import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    build_model_runner,
    extract_chunk_kv,
    load_tokenizer,
    load_working_set_files,
    prefill_and_get_slots,
    write_chunk_bin,
    escape_slot_id,
)

logger = logging.getLogger(__name__)

DIRECTION_A_V6_VERDICT_PREAMBLE = (
    "# Repository: pandas-dev/pandas\n"
    "# Working set context for coding-agent serving. "
    "The following file is part of this repository.\n"
    "\n"
    "You are a senior code reviewer.\n"
    "\n"
    "## Agent role\n"
    "ROLE\n"
    "\n"
    "## Case\n"
    "CASE\n"
    "\n"
    "## Instruction\n"
    "Inspect the repeated repository code. Decide if the code needs a fix "
    "(any non-trivial bug, risk, or missing handling) or if it is clean as-is.\n"
    "\n"
    "## Upstream context\n"
    "UPSTREAM\n"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--working-set-manifest", required=True)
    ap.add_argument("--pandas-src", required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--max-files", type=int, default=25)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("[Dir A v6 verdict-aligned] args=%s", vars(args))

    preamble = DIRECTION_A_V6_VERDICT_PREAMBLE
    preamble_sha1 = hashlib.sha1(preamble.encode("utf-8")).hexdigest()
    out_dir = Path(args.out_root) / args.run_tag
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (out_dir / "preamble.txt").write_text(preamble, encoding="utf-8")

    logger.info("[Dir A v6 verdict-aligned] using verdict-anchored preamble (%d chars):", len(preamble))
    for line in preamble.split("\n"):
        logger.info("  | %s", line)
    logger.info("[Dir A v6 verdict-aligned] preamble_sha1: %s", preamble_sha1)

    server_args, model_config, model_runner = build_model_runner(
        args.model_path, args.tp_size, args.mem_fraction_static
    )
    tokenizer = load_tokenizer(server_args)
    kv_pool = model_runner.token_to_kv_pool
    layer_num = model_config.num_hidden_layers
    head_num = kv_pool.head_num
    head_dim = kv_pool.head_dim
    dtype = kv_pool.dtype

    work_set = load_working_set_files(args.working_set_manifest, args.pandas_src, args.max_files)
    logger.info("[Dir A v6 verdict-aligned] %d files in working set", len(work_set))

    slot_text_to_kv = prefill_and_get_slots(
        model_runner, tokenizer, preamble,
        case="CASE", upstream="UPSTREAM", role="ROLE",  # placeholders substituted at runtime by server
    )
    # The 3 placeholders must be tuned per agent_role + case; we store
    # per-precompute call (server does substitution); for now record
    # per (file, role) extraction:
    succeeded = 0
    skipped = 0
    chunk_index_meta: list[dict] = []
    for fi, (rel_path, abs_path) in enumerate(work_set):
        text = abs_path.read_text(encoding="utf-8", errors="replace")[:20000]
        meta, kv_tensors = extract_chunk_kv(model_runner, tokenizer, preamble,
                                            file_text=text, rel_path=rel_path)
        if meta is None:
            skipped += 1
            continue
        out_bin = write_chunk_bin(out_dir, meta, kv_tensors)
        chunk_index_meta.append({**meta, "path": str(out_bin.relative_to(out_dir.parent))})
        succeeded += 1
        if (fi + 1) % 5 == 0:
            logger.info("[Dir A v6 verdict-aligned] chunk %d/%d done; storage=%d MB so far",
                        fi + 1, len(work_set),
                        sum((out_dir / c["path"]).stat().st_size for c in chunk_index_meta) // (1024*1024))
    (out_dir / "manifest.json").write_text(json.dumps({
        "run_tag": args.run_tag,
        "preamble_sha1": preamble_sha1,
        "preamble_chars": len(preamble),
        "layer_num": layer_num,
        "head_num": head_num,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "chunks_total": len(work_set),
        "chunks_succeeded": succeeded,
        "chunks_skipped": skipped,
        "preamble": preamble,
    }, indent=2))
    (out_dir / "chunks_index.jsonl").write_text("\n".join(json.dumps(c) for c in chunk_index_meta) + "\n")
    logger.info("[Dir A v6 verdict-aligned] DONE: %d chunks stored, %d skipped", succeeded, skipped)


if __name__ == "__main__":
    main()
