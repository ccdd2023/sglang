from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected one match in {path}, found {count}: {old!r}"
        )
    path.write_text(text.replace(old, new, 1))


schedule_batch = ROOT / "python/sglang/srt/managers/schedule_batch.py"
replace_once(
    schedule_batch,
    "from sglang.srt.mem_cache.approx_kv.request import "
    "parse_request_metadata\n"
    "from sglang.srt.mem_cache.memory_pool import ReqToTokenPool\n",
    "from sglang.srt.mem_cache.approx_kv.request import "
    "parse_request_metadata\n"
    "from sglang.srt.mem_cache.approx_kv.runtime import "
    "restore_request_prefix\n"
    "from sglang.srt.mem_cache.memory_pool import ReqToTokenPool\n",
)
replace_once(
    schedule_batch,
    "            else:\n"
    "                self.cache_protected_len = len(self.prefix_indices)\n"
    "\n"
    "            if self.is_dllm():\n",
    "            else:\n"
    "                self.cache_protected_len = len(self.prefix_indices)\n"
    "\n"
    "            if self.approx_kv_metadata is not None:\n"
    "                restore_request_prefix(tree_cache, self)\n"
    "\n"
    "            if self.is_dllm():\n",
)

common = ROOT / "python/sglang/srt/mem_cache/common.py"
replace_once(
    common,
    "from sglang.srt.mem_cache.allocator.swa import "
    "SWATokenToKVPoolAllocator\n",
    "from sglang.srt.mem_cache.allocator.swa import "
    "SWATokenToKVPoolAllocator\n"
    "from sglang.srt.mem_cache.approx_kv.runtime import "
    "register_request_segments\n",
)
replace_once(
    common,
    "    effective_kv_committed_len = req.effective_kv_committed_len()\n"
    "    tree_cache.cache_finished_req(\n",
    "    effective_kv_committed_len = req.effective_kv_committed_len()\n"
    "    register_request_segments(tree_cache, req)\n"
    "    tree_cache.cache_finished_req(\n",
)
