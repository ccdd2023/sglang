#!/usr/bin/env python3
"""Offline precompute of a codebase's KV cache, AST-marked, serialized to disk.

This is Phase 1 of the "offline-precomputed codebase KV -> CPU pool ->
prefetch -> lossy reuse" feature (see
``~/.claude/plans/codebase-kv-cache-ast-cpu-memory-kv-gpu-enchanted-dragonfly.md``).

For each file in a working set it:
  1. Renders ``preamble + "\\n" + file_text`` as a prefill prompt.
  2. Runs a prefill-only forward via the in-process ``ModelRunner`` (direct
     access to the GPU ``MHATokenToKVPool`` — no network server).
  3. AST-chunks the file text (``ASTChunker``), maps chunk byte offsets to
     token offsets, and extracts each chunk's per-layer K/V tensors from
     ``kv_pool.k_buffer[layer][out_cache_loc]``.
  4. Serializes one ``.bin`` per chunk plus a ``manifest.jsonl`` under
     ``results/codebase_kv/<run_tag>/``.

The serialized dir is loaded at server start by
``python/sglang/srt/mem_cache/codebase_kv_loader.py`` (Phase 2) into a
dedicated CPU host pool, and the read path (Phase 3) transfers chunks
CPU->GPU on reuse.

Honest accuracy bound: precompute does NOT fix cross-context KV loss (raw
copy + RoPE across a different prefix is lossy; proven F1 0.46@1400tok,
0.00@7100tok). The ``preamble`` (canonical shared prefix) is the only
losslessly-reusable part. File content at shifted positions stays lossy.

Usage:
    python scripts/precompute_codebase_kv.py \\
        --model-path Qwen/Qwen2.5-Coder-7B-Instruct \\
        --working-set-manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \\
        --pandas-src results/giant_codebase/pandas_src \\
        --run-tag pandas_5case \\
        --max-files 25

Output:
    results/codebase_kv/<run_tag>/
        manifest.jsonl
        chunks/<slot_id_escaped>__<sig>.bin
        preamble.txt
        preamble_token_ids.json
        meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Project rule: experiment output goes to results/, never /tmp.
DEFAULT_OUT_ROOT = "results/codebase_kv"

# Canonical shared prefix. Reproduced byte-exact at the start of every task
# prompt (prepended to the system message) so the preamble's KV is a true
# prefix match -> losslessly radix-cacheable AND a lossless prefix of every
# file's stored span. File content at shifted positions stays lossy.
DEFAULT_PREAMBLE = (
    "# Repository: pandas-dev/pandas\n"
    "# Working set context for coding-agent serving. "
    "The following file is part of this repository.\n"
)


# --------------------------------------------------------------------------- #
# Working-set file selection.
# --------------------------------------------------------------------------- #
# Mirrors benchmark/multi_workflow/swesmith_pandas_loader.py's git-diff parser
# so the precompute set matches what the giant-codebase driver feeds the server.
_GIT_DIFF_RE = re.compile(r"^diff --git a/(?P<path>\S+) b/\S+$")


def parse_patch_files(patch_text: str) -> List[str]:
    """Extract affected .py file paths from a unified diff (dedup, first-seen)."""
    seen = set()
    files: List[str] = []
    for line in patch_text.splitlines():
        m = _GIT_DIFF_RE.match(line)
        if not m:
            continue
        path = m.group("path")
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def load_working_set_files(
    task_manifest: str,
    pandas_src: str,
    max_files: int,
    segment_count: int = 5,
    sibling_window: int = 4,
) -> List[Tuple[str, str]]:
    """Return ``[(rel_path, text), ...]`` for the working set.

    Mirrors ``bench_giant_codebase_reuse.build_segments_for_task`` exactly:
    patched files first (in patch order), then same-directory sibling .py
    files (sorted), collecting ``segment_count + sibling_window`` candidates
    per task and taking the first ``segment_count``. Each file truncated to
    ``max_file_chars`` (8000, the driver default). Files de-duplicated across
    tasks; collection stops at ``max_files``.
    """
    max_file_chars = 8000
    src_root = Path(pandas_src)
    seen: set = set()
    out: List[Tuple[str, str]] = []

    def load_text(rel: str) -> Optional[str]:
        p = src_root / rel
        if not p.is_file() or not p.name.endswith(".py"):
            return None
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("skip %s: %s", rel, e)
            return None
        if max_file_chars and len(text) > max_file_chars:
            text = text[:max_file_chars].rstrip()
        if not text.strip():
            return None
        return text

    with open(task_manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            patch = rec.get("patch") or ""
            patched = parse_patch_files(patch)
            # Build the ordered candidate list for THIS task exactly like the
            # driver: patched files, then siblings up to seg+sib cap.
            ordered: List[str] = list(patched)
            if sibling_window > 0 and len(ordered) < segment_count:
                for rel in list(patched):
                    anchor_dir = (src_root / rel).parent
                    if not anchor_dir.is_dir():
                        continue
                    for sibling in sorted(anchor_dir.iterdir()):
                        if not sibling.is_file() or not sibling.name.endswith(".py"):
                            continue
                        sib_rel = str(sibling.relative_to(src_root))
                        if sib_rel in ordered:
                            continue
                        ordered.append(sib_rel)
                        if len(ordered) >= segment_count + sibling_window:
                            break
                    if len(ordered) >= segment_count + sibling_window:
                        break
            # Take the first segment_count candidates that load + de-dup.
            for rel in ordered[:segment_count]:
                if rel in seen:
                    continue
                text = load_text(rel)
                if text is None:
                    continue
                seen.add(rel)
                out.append((rel, text))
                if len(out) >= max_files:
                    return out
            if len(out) >= max_files:
                break
    return out


# --------------------------------------------------------------------------- #
# ModelRunner setup (verified API: bench_one_batch.py:extend).
# --------------------------------------------------------------------------- #
class TreeCacheNamespace(SimpleNamespace):
    """Minimal tree_cache stand-in (no prefix caching, just allocation).

    Mirrors python/sglang/bench_one_batch.py:377 so we can drive
    ScheduleBatch.init_new / prepare_for_extend without a full RadixCache.
    """

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def evict(self, params: Any) -> None:
        pass


def build_model_runner(model_path: str, tp_size: int, mem_fraction_static: float):
    """Construct a standalone ModelRunner for prefill-only inference."""
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs

    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        port=30000,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_piecewise_cuda_graph=True,
    )
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=tp_size,
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    return server_args, model_config, model_runner


def load_tokenizer(server_args):
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    return get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )


@torch.inference_mode()
def prefill_and_get_slots(
    model_runner, model_config, token_ids: List[int]
) -> torch.Tensor:
    """Run one prefill-only forward; return the ``out_cache_loc`` tensor.

    ``out_cache_loc[i]`` is the KV-pool slot index that token ``i`` was
    written to (verified path: alloc_for_extend -> ScheduleBatch ->
    ForwardBatch.init_new). The caller slices ``kv_pool.{k,v}_buffer[layer]``
    by these indices.
    """
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool  # noqa: F401
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    req = Req(
        rid=0,
        origin_input_text="",
        origin_input_ids=list(token_ids),
        sampling_params=SamplingParams(max_new_tokens=0),
    )
    req.fill_ids = req.origin_input_ids
    req.logprob_start_len = -1
    req.set_extend_input_len(len(req.fill_ids))

    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=[req],
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    forward_batch = ForwardBatch.init_new(batch.get_model_worker_batch(), model_runner)
    model_runner.forward_extend(forward_batch)
    return forward_batch.out_cache_loc.detach()


# --------------------------------------------------------------------------- #
# KV extraction + serialization.
# --------------------------------------------------------------------------- #
def escape_slot_id(slot_id: str) -> str:
    """Filesystem-safe encoding of a slot_id for the .bin filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", slot_id)


def extract_chunk_kv(
    kv_pool,
    out_cache_loc: torch.Tensor,
    tok_start: int,
    tok_end: int,
    layer_num: int,
) -> torch.Tensor:
    """Extract one chunk's KV as a contiguous CPU tensor.

    Layout: ``[2, layer_num, chunk_len, head_num, head_dim]`` (K then V),
    matching the ``layer_first`` layout of ``MHATokenToKVPoolHost`` so the
    Phase 2 loader can ``readinto`` the host buffer directly.
    """
    slots = out_cache_loc[tok_start:tok_end]
    # gather on GPU then move to CPU once
    k_layers = torch.stack(
        [kv_pool.k_buffer[layer_id][slots] for layer_id in range(layer_num)]
    )  # [L, chunk_len, head_num, head_dim]
    v_layers = torch.stack(
        [kv_pool.v_buffer[layer_id][slots] for layer_id in range(layer_num)]
    )
    kv = torch.stack([k_layers, v_layers])  # [2, L, chunk_len, H, D]
    return kv.contiguous().cpu()


def write_chunk_bin(path: Path, kv_tensor: torch.Tensor) -> None:
    """Serialize a KV tensor as raw fp16/bf16 bytes (tofile idiom)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kv_tensor.view(torch.uint8).numpy().tofile(str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument(
        "--working-set-manifest",
        required=True,
        help="SWE-Smith task manifest.jsonl (results/giant_codebase/.../manifest.jsonl)",
    )
    ap.add_argument(
        "--pandas-src", required=True, help="pandas checkout root (results/giant_codebase/pandas_src)"
    )
    ap.add_argument("--run-tag", default="latest")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--max-files", type=int, default=25)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.8)
    ap.add_argument("--preamble", default=None, help="override canonical preamble text")
    ap.add_argument(
        "--no-canonical-prefix",
        action="store_true",
        help="omit preamble (pure lossy ablation; preamble file still written empty)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from sglang.srt.mem_cache.ast_chunker import ASTChunker
    from sglang.srt.mem_cache.radix_cache import byte_to_token_offset

    # Force L4 toplevel chunking (function/class boundaries, matching the
    # server's production read path with SGLANG_CHUNK_TOPLEVEL=1).
    os.environ.setdefault("SGLANG_CHUNK_TOPLEVEL", "1")
    # Coarse OFF: we want per-chunk granularity so partial-share works.
    os.environ.setdefault("SGLANG_CHUNK_COARSE", "0")
    chunker = ASTChunker()

    preamble = "" if args.no_canonical_prefix else (args.preamble or DEFAULT_PREAMBLE)
    preamble_sha1 = hashlib.sha1(preamble.encode("utf-8")).hexdigest()

    out_dir = Path(args.out_root) / args.run_tag
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (out_dir / "preamble.txt").write_text(preamble, encoding="utf-8")

    logger.info("loading model %s (tp=%d)...", args.model_path, args.tp_size)
    server_args, model_config, model_runner = build_model_runner(
        args.model_path, args.tp_size, args.mem_fraction_static
    )
    tokenizer = load_tokenizer(server_args)
    kv_pool = model_runner.token_to_kv_pool
    layer_num = model_config.num_hidden_layers
    head_num = kv_pool.head_num
    head_dim = kv_pool.head_dim
    dtype = str(kv_pool.store_dtype) if hasattr(kv_pool, "store_dtype") else str(kv_pool.k_buffer[0].dtype)
    # Layout the server will use; default layer_first (MHA standard).
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
        "created": int(time.time()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("meta: %s", meta)

    files = load_working_set_files(args.working_set_manifest, args.pandas_src, args.max_files)
    logger.info("working set: %d files", len(files))

    preamble_token_ids: List[int] = []
    if preamble:
        preamble_token_ids = tokenizer.encode(preamble, add_special_tokens=False)
    (out_dir / "preamble_token_ids.json").write_text(json.dumps(preamble_token_ids))

    manifest_path = out_dir / "manifest.jsonl"
    n_chunks = 0
    with manifest_path.open("w") as mf:
        for fi, (rel, text) in enumerate(files):
            slot_id = f"code_base:{rel}"
            prompt_text = (preamble + "\n" + text) if preamble else text
            token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(token_ids) < 2:
                continue
            try:
                out_cache_loc = prefill_and_get_slots(model_runner, model_config, token_ids)
            except Exception as e:
                logger.warning("prefill failed for %s: %s", rel, e)
                continue

            # Chunk the FILE text (not the preamble). Byte offsets are within
            # ``text``; token offsets are within the full prompt, offset by the
            # preamble token count.
            chunks = chunker.chunk_text(text)
            if not chunks:
                logger.info("[%d/%d] %s: no AST chunks, skipping", fi + 1, len(files), rel)
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
                }
                mf.write(json.dumps(rec) + "\n")
                n_chunks += 1
            logger.info(
                "[%d/%d] %s: %d chunks (%d toks)",
                fi + 1, len(files), rel, len(chunks), len(token_ids),
            )
            # Free the radix slots we allocated for this file so the next
            # prefill has pool space (the dummy tree_cache never evicts).
            try:
                model_runner.token_to_kv_pool_allocator.free(out_cache_loc)
            except Exception:
                pass

    logger.info("done: %d chunks -> %s/manifest.jsonl", n_chunks, out_dir)


if __name__ == "__main__":
    main()
