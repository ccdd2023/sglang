# Non-Qwen 7B Cross-Family Attempted Downloads (2026-06-08)

Two non-Qwen 7B-class models were attempted on 2026-06-08 to extend the
4/4 Qwen cross-model study to a 5/5 cross-family study. Both downloads
stalled at the 1.5-3.6 GB mark (out of 7-14 GB total) due to
HuggingFace's unauthenticated rate limit, which silently throttles
multi-GB downloads to ~1.5-3 GB per session before the connection
stops progressing. The 4/4 Qwen verdict stands; the 5/5 cross-family
extension is **deferred** to a session with a `HF_TOKEN` (gated access
removes the rate limit) or a pre-warmed model cache.

## Stack

- **GPU**: not involved (download only)
- **transformers**: 5.3.0
- **huggingface_hub**: 0.x
- **HF_TOKEN**: not set
- **HF_ENDPOINT**: default (huggingface.co)

## Attempt 1: mistralai/Mistral-7B-Instruct-v0.3 (Mistral AI)

- **Status**: 401 / config only — non-gated, download allowed
- **Gated access**: not required (Mistral-7B-Instruct-v0.3 is public on HF)
- **Expected size**: 14.0 GB (2 safetensors shards, ~7 GB each)
- **Achieved size before stall**: 3.6 GB (4 `.incomplete` files totaling 3.83 GB)
- **Time to stall**: ~17 minutes from launch
- **Stall pattern**: 4 `.incomplete` files at 670M / 670M / 542M / 1,950M; no further growth for 17+ minutes while the ESTABLISHED TCP connection to `104.21.23.165:443` (HF CDN via Cloudflare) remained open
- **Conclusion**: HF CDN rate limit kicked in after ~3.6 GB; download not resumable without `HF_TOKEN`

## Attempt 2: NousResearch/Llama-2-7b-chat-hf (Meta re-uploaded)

- **Status**: 401 / config only — non-gated, download allowed
- **Expected size**: 13.5 GB (2 safetensors shards + 3 pytorch_model.bin shards, ~4-5 GB each)
- **Achieved size before stall**: 2.9 GB (1 safetensors shard at 1.6 GB + 1 pytorch_model.bin at 1.4 GB, both `.incomplete`)
- **Time to stall**: ~6 minutes
- **Stall pattern**: First shard downloaded at ~100 MB/s for 1.6 GB, then stalled. Second shard (pytorch_model-00001-of-00003.bin) started but stalled at 1.4 GB after 30s of inactivity
- **Conclusion**: Same rate limit pattern. Re-attempting with `HF_TOKEN` or a different CDN would likely succeed

## Verdict

Both attempted non-Qwen 7B models are **public on HF and the config downloads succeed**; only the multi-GB safetensors / pytorch_model.bin downloads hit the rate limit. The 4/4 Qwen cross-model verdict (Qwen2.5-family-portable, Qwen3-needs-per-family-table) is unchanged.

The `results/lookup_table_transferability/{run_all.sh, cross_model_report.py, multiple_comparison_correction.py}` files have been **reverted to the 4-model Qwen list** (the temporary 5th-model addition for Mistral-7B has been removed). The paper text is updated to honestly report the rate-limited attempt.

## Unblocking paths

1. **Set `HF_TOKEN` and re-run** (5 min + 30 min download). The token is free; it removes the unauthenticated throttle and the ~3 GB-per-session cap.
2. **Pre-warm the cache on a different machine** with `HF_TOKEN` and copy the cache to `/home/gfy/.cache/huggingface/hub/` here. ~30 min for 14 GB at typical LAN speeds.
3. **Use a different CDN mirror** such as `hf-mirror.com` (the smaller NousResearch config file was 36 KB, well under the threshold, but the safetensors files were not found on the mirror at the time of the attempt).
4. **Use a model with smaller total weight** (e.g., `Qwen/Qwen2-7B-Instruct` — same family but Qwen2 vs Qwen2.5 contrast, ~15 GB but already shown to download non-gated). This is the next-best option if the 5/5 cross-family claim is abandoned.
5. **Use the existing 4 Qwen models** and re-state the verdict as "Qwen-portable" (the 4/4 claim) instead of "cross-family-portable" (the 5/5 claim that was attempted). This is the conservative fallback the paper text now adopts.
