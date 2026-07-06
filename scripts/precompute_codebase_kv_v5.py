#!/usr/bin/env python3
"""Direction A v4 — Per-role extraction (Round 6, 2026-07-02).

Uses Direction A v3 preamble PLUS actual AGENT_ROLES values instead
of the "ROLE" placeholder. This narrows the prefix gap to JUST
case_id value (~30 tokens of CASE mismatch) + upstream_context
value.

Per-agent F1 should be much higher when the agent's actual role matches
the pool's role. Other agents fall back to baseline behavior.

5 pools extracted (one per role): implementer, debugger, verifier,
auditor, optimizer.

Usage:
    python scripts/precompute_codebase_kv_v5.py \\
        --model-path Qwen/Qwen2.5-Coder-7B-Instruct \\
        --working-set-manifest ... \\
        --pandas-src ... \\
        --run-tag pandas_5case_v5_role_implementer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
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

# Same as Direction A v3, but ROLE replaced with actual values
def make_preamble(role: str) -> str:
    return (
        "# Repository: pandas-dev/pandas\n"
        "# Working set context for coding-agent serving. "
        "The following file is part of this repository.\n"
        "\n"
        "You are a senior software engineering agent.\n"
        "\n"
        f"## Agent role\n"
        f"{role}\n"
        "\n"
        "## Case\n"
        "CASE\n"
        "\n"
        "## Instruction\n"
        "Inspect the repeated repository code and answer with one concise implementation risk.\n"
        "\n"
        "## Upstream context\n"
        "UPSTREAM\n"
    )

# Match the bench's AGENT_ROLES list
AGENT_ROLES = ["implementer", "debugger", "verifier", "auditor", "optimizer"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--working-set-manifest", required=True)
    ap.add_argument("--pandas-src", required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--role", default="implementer", help="agent role to bake into preamble")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--max-files", type=int, default=25)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from sglang.srt.mem_cache.ast_chunker import ASTChunker
    from sglang.srt.mem_cache.radix_cache import byte_to_token_offset

    os.environ.setdefault("SGLANG_CHUNK_TOPLEVEL", "1")
    os.environ.setdefault("SGLANG_CHUNK_COARSE", "0")

    preamble = make_preamble(args.role)
    preamble_sha1 = hashlib.sha1(preamble.encode("utf-8")).hexdigest()
    out_dir = Path(args.out_root) / args.run_tag
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (out_dir / "preamble.txt").write_text(preamble, encoding="utf-8")

    logger.info("[Dir A v4] role=%s, preamble (%d chars):", args.role, len(preamble))
    for line in preamble.split("\n"):
        logger.info("  | %s", line)
    logger.info("[Dir A v4] preamble_sha1: %s", preamble_sha1)

    server_args, model_config, model_runner = build_model_runner(
        args.model_path, args.tp_size, args.mem_fraction_static
    )
    tokenizer = load_tokenizer(server_args)
    kv_pool = model_runner.token_to_kv_pool
    layer_num = model_config.num_hidden_layers
    head_num = kv_pool.head_num
    head_dim = kv_pool.head_dim
    dtype = str(kv_pool.store_dtype) if hasattr(kv_pool, "store_dtype") else str(kv_pool.k_buffer[0].dtype)
    layout = os.environ.get("SGLANG_PRECOMPUTE_HOST_LAYOUT", "layer_first")

    meta = {
        "model_name": args.model_path,
        "tp_size": args.tp_size,
        "head_num": int(head_num),
        "head_dim": int(head_dim),
        "layer_num": int(layer_num),
        "dtype": dtype,
        "layout": layout,
        "preamble_sha1": preamble_sha1,
        "preamble_len": len(preamble),
        "preamble_variant": f"direction_a_v4_role_{args.role}",
        "role": args.role,
        "created": int(time.time()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    files = load_working_set_files(args.working_set_manifest, args.pandas_src, args.max_files)
    logger.info("working set: %d files", len(files))

    preamble_token_ids = tokenizer.encode(preamble, add_special_tokens=False)
    (out_dir / "preamble_token_ids.json").write_text(json.dumps(preamble_token_ids))
    logger.info("preamble token count: %d", len(preamble_token_ids))

    chunker = ASTChunker()
    manifest_path = out_dir / "manifest.jsonl"
    n_chunks = 0
    with manifest_path.open("w") as mf:
        for fi, (rel, text) in enumerate(files):
            slot_id = f"code_base:{rel}"
            prompt_text = preamble + "\n" + text
            token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(token_ids) < 2:
                continue
            try:
                out_cache_loc = prefill_and_get_slots(model_runner, model_config, token_ids)
            except Exception as e:
                logger.warning("prefill failed for %s: %s", rel, e)
                continue
            chunks = chunker.chunk_text(text)
            if not chunks:
                continue
            preamble_tok_len = len(preamble_token_ids)
            for chunk in chunks:
                tok_start = preamble_tok_len + byte_to_token_offset(text, chunk.byte_start, tokenizer)
                tok_end = preamble_tok_len + byte_to_token_offset(text, chunk.byte_end, tokenizer)
                if tok_end <= tok_start:
                    continue
                chunk_token_ids = token_ids[tok_start:tok_end]
                if len(chunk_token_ids) == 0:
                    continue
                kv_tensor = extract_chunk_kv(
                    kv_pool, out_cache_loc, tok_start, tok_end, layer_num
                )
                bin_name = f"{escape_slot_id(slot_id)}__{chunk.signature}.bin"
                write_chunk_bin(out_dir / "chunks" / bin_name, kv_tensor)
                rec = {
                    "slot_id": slot_id,
                    "chunk_signature": chunk.signature,
                    "anchor_type": chunk.anchor_type,
                    "name": chunk.name,
                    "byte_start": int(chunk.byte_start),
                    "byte_end": int(chunk.byte_end),
                    "start_token": int(tok_start - preamble_tok_len),
                    "end_token": int(tok_end - preamble_tok_len),
                    "n_tokens": int(tok_end - tok_start),
                    "token_ids": list(chunk_token_ids),
                    "bin_path": f"chunks/{bin_name}",
                    "preamble_sha1": preamble_sha1,
                    "role": args.role,
                }
                mf.write(json.dumps(rec) + "\n")
                n_chunks += 1
            logger.info(
                "[%d/%d] %s: %d chunks (%d toks)",
                fi + 1, len(files), rel, len(chunks), len(token_ids),
            )
            try:
                model_runner.token_to_kv_pool_allocator.free(out_cache_loc)
            except Exception:
                pass

    logger.info("[Dir A v4 %s] done: %d chunks -> %s/manifest.jsonl",
                args.role, n_chunks, out_dir)


if __name__ == "__main__":
    main()