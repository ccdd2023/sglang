from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old!r}")
    path.write_text(text.replace(old, new, 1))


radix_cache = ROOT / "python/sglang/srt/mem_cache/radix_cache.py"
replace_once(
    radix_cache,
    "from sglang.srt.mem_cache.cache_init_params import CacheInitParams\n",
    "from sglang.srt.mem_cache.approx_kv.config import "
    "ApproxKVFeatureConfig\n"
    "from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager\n"
    "from sglang.srt.mem_cache.cache_init_params import CacheInitParams\n",
)
replace_once(
    radix_cache,
    "        self.evictable_leaves = set()\n        self.reset()\n",
    "        self.evictable_leaves = set()\n"
    "        self.approx_kv = ApproxKVManager(\n"
    "            ApproxKVFeatureConfig.from_env()\n"
    "        )\n"
    "        self.reset()\n",
)
replace_once(
    radix_cache,
    "        self.evictable_leaves.clear()\n"
    "        self._reset_session_radix_state()\n",
    "        self.evictable_leaves.clear()\n"
    "        self.approx_kv.reset()\n"
    "        self._reset_session_radix_state()\n",
)

schedule_batch = ROOT / "python/sglang/srt/managers/schedule_batch.py"
replace_once(
    schedule_batch,
    "from sglang.srt.mem_cache.memory_pool import ReqToTokenPool\n",
    "from sglang.srt.mem_cache.approx_kv.request import "
    "parse_request_metadata\n"
    "from sglang.srt.mem_cache.memory_pool import ReqToTokenPool\n",
)
replace_once(
    schedule_batch,
    "        self.sampling_params = sampling_params\n"
    "        self.custom_logit_processor = custom_logit_processor\n",
    "        self.sampling_params = sampling_params\n"
    "        self.approx_kv_metadata = parse_request_metadata(\n"
    "            sampling_params.custom_params\n"
    "            if isinstance(sampling_params.custom_params, dict)\n"
    "            else None\n"
    "        )\n"
    "        if self.approx_kv_metadata is not None:\n"
    "            self.approx_kv_metadata.validate_prompt_length(\n"
    "                len(self.origin_input_ids)\n"
    "            )\n"
    "        self.custom_logit_processor = custom_logit_processor\n",
)
replace_once(
    schedule_batch,
    "        self.skip_radix_cache_insert = bootstrap_host == FAKE_BOOTSTRAP_HOST\n",
    "        self.skip_radix_cache_insert = (\n"
    "            bootstrap_host == FAKE_BOOTSTRAP_HOST\n"
    "            or self.approx_kv_metadata is not None\n"
    "        )\n",
)
