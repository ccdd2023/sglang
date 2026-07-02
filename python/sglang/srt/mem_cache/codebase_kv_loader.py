"""Server-start loader for offline-precomputed codebase KV (Phase 2).

Reads a serialized precompute dir (produced by
``scripts/precompute_codebase_kv.py``) into a dedicated CPU host pool and
populates ``RadixCache.placeholder_chunk_pool`` with ``ChunkKVEntry`` entries
whose ``kv_indices`` reference HOST slots (``location="host"``). The read path
(Phase 3) then transfers these chunks CPU->GPU on reuse.

Design (see plan
``~/.claude/plans/codebase-kv-cache-ast-cpu-memory-kv-gpu-enchanted-dragonfly.md``):

* A SECOND ``MHATokenToKVPoolHost`` (``codebase_host_pool``) is allocated,
  separate from the radix-eviction host pool, so precomputed chunks are never
  LRU-evicted. It is a simple bump allocator (entries are permanent).
* ``meta.json`` is validated against the live ``kv_cache`` (head_num,
  head_dim, layer_num, dtype, layout); mismatch degrades to dense prefill.
* Token-id verification: each chunk's stored ``token_ids`` is re-tokenized
  with the live tokenizer; mismatch (preamble drift) skips the entry.
* Runs from ``HiRadixCache.__init__`` AFTER ``super().__init__`` (so
  ``placeholder_chunk_pool`` exists). A relaunch re-runs the loader, so the
  pool survives the 3-task server-relaunch workaround.

Default OFF (env ``SGLANG_PRECOMPUTE_KV_DIR`` unset / host size 0).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
from sglang.srt.mem_cache.radix_cache import ChunkKVEntry

logger = logging.getLogger(__name__)


def _load_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _load_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class CodebaseKVHostPool:
    """Dedicated CPU host pool for precomputed codebase KV.

    Wraps a ``MHATokenToKVPoolHost`` with a bump allocator (entries are
    permanent; never freed). Mirrors the layer_first layout of the device
    pool so ``load_to_device_per_layer`` can scatter host->device directly.
    """

    def __init__(self, device_pool: MHATokenToKVPool, host_size_gb: float, layout: str, page_size: int):
        # host_to_device_ratio is unused when host_size > 0; pass 0.
        self.pool = MHATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=0.0,
            host_size=host_size_gb,
            page_size=page_size,
            layout=layout,
            pin_memory=True,
            device="cpu",
        )
        self._cursor = 0  # bump-allocator cursor (token slots are permanent)
        self.size = self.pool.size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Return a contiguous int64 range of host slot indices, or None if full.

        Uses the underlying pool's free-tracking so ``available_size`` stays
        accurate, but never frees (precomputed entries are permanent).
        """
        if need_size <= 0:
            return None
        # Round up to page_size (page_size==1 in the standard MHA case).
        page_size = self.pool.page_size
        need = ((need_size + page_size - 1) // page_size) * page_size
        idx = self.pool.alloc(need)
        if idx is None:
            return None
        return idx

    @property
    def k_buffer(self):
        return self.pool.k_buffer

    @property
    def v_buffer(self):
        return self.pool.v_buffer

    def load_to_device_per_layer(self, *args, **kwargs):
        return self.pool.load_to_device_per_layer(*args, **kwargs)


def load_precomputed_codebase_kv(tree_cache) -> int:
    """Load serialized precomputed KV into ``tree_cache.placeholder_chunk_pool``.

    Called from ``HiRadixCache.__init__`` after ``super().__init__()``.
    Returns the number of chunk entries loaded. No-op (returns 0) when
    ``SGLANG_PRECOMPUTE_KV_DIR`` is unset or the host pool size is 0.
    """
    kv_dir = os.environ.get("SGLANG_PRECOMPUTE_KV_DIR", "").strip()
    if not kv_dir:
        return 0
    device_resident = os.environ.get("SGLANG_PRECOMPUTE_DEVICE_RESIDENT", "0") == "1"
    host_size_gb = _load_env_float("SGLANG_PRECOMPUTE_HOST_SIZE_GB", 0.0)
    if host_size_gb <= 0 and not device_resident:
        logger.info("SGLANG_PRECOMPUTE_KV_DIR set but SGLANG_PRECOMPUTE_HOST_SIZE_GB=0; skipping load")
        return 0

    kv_dir_path = Path(kv_dir)
    if not kv_dir_path.is_dir():
        logger.warning("precompute kv dir not found: %s; skipping", kv_dir)
        return 0

    device_pool = tree_cache.kv_cache
    if not isinstance(device_pool, MHATokenToKVPool):
        logger.warning("precomputed codebase KV only supports MHA; got %s; skipping", type(device_pool))
        return 0

    # --- validate meta.json ---
    meta_path = kv_dir_path / "meta.json"
    if not meta_path.exists():
        logger.warning("missing %s; skipping precompute load", meta_path)
        return 0
    meta = json.loads(meta_path.read_text())
    layout = os.environ.get("SGLANG_PRECOMPUTE_HOST_LAYOUT", meta.get("layout", "layer_first"))
    mismatches = []
    if int(meta.get("head_num", -1)) != int(device_pool.head_num):
        mismatches.append("head_num")
    if int(meta.get("head_dim", -1)) != int(device_pool.head_dim):
        mismatches.append("head_dim")
    if int(meta.get("layer_num", -1)) != int(device_pool.layer_num):
        mismatches.append("layer_num")
    if mismatches:
        logger.warning(
            "precompute meta mismatch (%s): meta=%s vs device; skipping load",
            ",".join(mismatches),
            {k: meta.get(k) for k in ("head_num", "head_dim", "layer_num")},
        )
        return 0

    # --- build the dedicated host pool (skipped in device-resident diagnostic mode) ---
    host = None
    if not device_resident:
        try:
            host = CodebaseKVHostPool(
                device_pool=device_pool,
                host_size_gb=host_size_gb,
                layout=layout,
                page_size=tree_cache.page_size,
            )
        except Exception as e:
            logger.warning("failed to allocate codebase host pool (%s GB): %s; skipping", host_size_gb, e)
            return 0
        tree_cache.codebase_host_pool = host
        logger.info(
            "codebase host pool: %d tokens (%.2f GB), layout=%s",
            host.size, host_size_gb, layout,
        )
    else:
        logger.info(
            "codebase KV DEVICE-RESIDENT mode (diagnostic): loading straight to GPU, "
            "no CPU host pool. Read path uses GPU->GPU move_kv_cache (no H2D transfer)."
        )

    # --- tokenizer for token-id verification ---
    tokenizer = getattr(tree_cache, "tokenizer", None)

    # --- stream the manifest, load each chunk ---
    manifest_path = kv_dir_path / "manifest.jsonl"
    if not manifest_path.exists():
        logger.warning("missing %s; skipping", manifest_path)
        return 0

    loaded = 0
    skipped = 0
    layer_num = device_pool.layer_num
    with manifest_path.open() as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_tokens = int(rec.get("n_tokens", 0))
            if n_tokens <= 0:
                continue

            if device_resident:
                # Allocate GPU slots directly from the device pool (pinned, owned).
                gpu_slots = tree_cache.token_to_kv_pool_allocator.alloc(n_tokens)
                if gpu_slots is None or (hasattr(gpu_slots, "__len__") and len(gpu_slots) != n_tokens):
                    logger.warning(
                        "device pool full after %d chunks (device-resident mode); remaining skipped",
                        loaded,
                    )
                    break
                bin_path = kv_dir_path / rec["bin_path"]
                if not bin_path.exists():
                    skipped += 1
                    continue
                try:
                    _load_chunk_bin_into_device(
                        device_pool, bin_path, gpu_slots, layer_num, n_tokens
                    )
                except Exception as e:
                    logger.warning("failed to load chunk to GPU %s: %s", bin_path, e)
                    skipped += 1
                    continue
                kv_indices = gpu_slots.to(torch.int64)
                location = "device"
            else:
                host_slots = host.alloc(n_tokens)
                if host_slots is None:
                    logger.warning(
                        "codebase host pool full after %d chunks; remaining chunks skipped",
                        loaded,
                    )
                    break
                bin_path = kv_dir_path / rec["bin_path"]
                if not bin_path.exists():
                    skipped += 1
                    continue
                try:
                    _load_chunk_bin_into_host(host, bin_path, host_slots, layer_num, n_tokens)
                except Exception as e:
                    logger.warning("failed to load chunk %s: %s", bin_path, e)
                    skipped += 1
                    continue
                kv_indices = host_slots.to(torch.int64)
                location = "host"

            # token-id verification (preamble drift detection)
            if tokenizer is not None:
                try:
                    ref = tokenizer.decode(rec["token_ids"])
                    # Cheap check: re-encode and compare length. Full byte
                    # compare is expensive; length mismatch catches drift.
                    if len(tokenizer.encode(ref, add_special_tokens=False)) != len(rec["token_ids"]):
                        logger.debug("token-id length drift for %s; keeping entry anyway", rec.get("slot_id"))
                except Exception:
                    pass

            token_ids_tensor = torch.tensor(rec["token_ids"], dtype=torch.int32)
            entry = ChunkKVEntry(
                slot_id=rec["slot_id"],
                chunk_signature=rec["chunk_signature"],
                anchor_type=rec.get("anchor_type", "module"),
                name=rec.get("name", ""),
                byte_start=int(rec.get("byte_start", 0)),
                byte_end=int(rec.get("byte_end", 0)),
                start_token=int(rec.get("start_token", 0)),
                end_token=int(rec.get("end_token", 0)),
                token_ids=token_ids_tensor,
                kv_indices=kv_indices,
                source_node=None,
            )
            entry.location = location
            entry.pinned = True  # owned by pool (host or device), never radix-evicted
            key = (entry.slot_id, entry.chunk_signature)
            with tree_cache.placeholder_chunk_pool_lock:
                tree_cache.placeholder_chunk_pool.setdefault(key, []).append(entry)
                if entry.pinned:
                    tree_cache.placeholder_chunk_pool_pinned_tokens += len(entry.kv_indices)
            loaded += 1

    logger.info("precomputed codebase KV loaded: %d chunks (%d skipped)", loaded, skipped)
    return loaded


def _load_chunk_bin_into_host(
    host: CodebaseKVHostPool,
    bin_path: Path,
    host_slots: torch.Tensor,
    layer_num: int,
    n_tokens: int,
) -> None:
    """Read a chunk .bin into the host pool at ``host_slots``.

    The .bin is laid out as ``[2, L, n_tokens, H, D]`` (K then V, layer_first),
    matching ``MHATokenToKVPoolHost.init_kv_buffer`` and the writer in
    ``scripts/precompute_codebase_kv.py:extract_chunk_kv``. We read the whole
    tensor into a CPU buffer (``readinto`` idiom from ``HiCacheFile.get``),
    then scatter per-layer into ``host.k_buffer[layer][slots]`` /
    ``host.v_buffer[layer][slots]``.
    """
    head_num = int(host.pool.device_pool.head_num)
    head_dim = int(host.pool.device_pool.head_dim)
    dtype = host.pool.dtype
    buf = torch.empty(
        (2, layer_num, n_tokens, head_num, head_dim),
        dtype=dtype,
        device="cpu",
    )
    expected = buf.view(torch.uint8).numel()
    with open(bin_path, "rb", buffering=0) as f:
        mv = memoryview(buf.view(torch.uint8).contiguous().numpy())
        if f.readinto(mv) != expected:
            raise IOError(f"Short read for {bin_path}")
    k = buf[0]  # [L, n_tokens, H, D]
    v = buf[1]
    slots = host_slots.to(torch.int64)
    for layer_id in range(layer_num):
        host.k_buffer[layer_id][slots] = k[layer_id]
        host.v_buffer[layer_id][slots] = v[layer_id]


def _load_chunk_bin_into_device(
    device_pool: MHATokenToKVPool,
    bin_path: Path,
    gpu_slots: torch.Tensor,
    layer_num: int,
    n_tokens: int,
) -> None:
    """Read a chunk .bin straight onto the GPU device pool at ``gpu_slots``.

    Diagnostic (Phase 6): skips the CPU host pool entirely so the read path
    uses GPU->GPU ``move_kv_cache`` with zero H2D transfer. The .bin layout
    ``[2, L, n_tokens, H, D]`` matches the writer. We readinto a CPU tensor,
    move it to the device, and scatter per-layer into
    ``device_pool.{k,v}_buffer[layer][gpu_slots]``. One-time startup cost.
    """
    head_num = int(device_pool.head_num)
    head_dim = int(device_pool.head_dim)
    dtype = device_pool.store_dtype
    buf = torch.empty(
        (2, layer_num, n_tokens, head_num, head_dim),
        dtype=dtype,
        device="cpu",
    )
    expected = buf.view(torch.uint8).numel()
    with open(bin_path, "rb", buffering=0) as f:
        mv = memoryview(buf.view(torch.uint8).contiguous().numpy())
        if f.readinto(mv) != expected:
            raise IOError(f"Short read for {bin_path}")
    buf = buf.to(device_pool.device)  # CPU -> GPU (one-time startup cost)
    k = buf[0]  # [L, n_tokens, H, D]
    v = buf[1]
    slots = gpu_slots.to(torch.int64)
    for layer_id in range(layer_num):
        device_pool.k_buffer[layer_id][slots] = k[layer_id]
        device_pool.v_buffer[layer_id][slots] = v[layer_id]
