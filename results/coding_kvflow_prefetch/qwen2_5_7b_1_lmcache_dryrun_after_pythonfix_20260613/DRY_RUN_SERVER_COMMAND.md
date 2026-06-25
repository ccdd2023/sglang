# Server Command Dry Run

No model server was launched.

```bash
/home/gfy/.conda/envs/sglang-kvflow-lmcache/bin/python -m sglang.launch_server --model-path /home/gfy/models/Qwen2.5-7B-Instruct --port 31341 --tp-size 1 --mem-fraction-static 0.78 --max-total-tokens 65536 --chunked-prefill-size 8192 --max-prefill-tokens 16384 --enable-cache-report --disable-cuda-graph --log-level error --enable-lmcache
```

- Baseline profile: `lmcache`
- LMCache config: `/home/gfy/CodeMAS_Project/sglang-kvflow/python/sglang/srt/mem_cache/storage/lmcache/example_config.yaml`
- LMCACHE_USE_EXPERIMENTAL: `True`
- HiCache storage backend: ``
- Hierarchical cache enabled: `False`
- Hierarchical cache suppressed for LMCache: `True`
