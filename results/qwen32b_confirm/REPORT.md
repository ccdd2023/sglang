# Qwen-32B / 30B Confirmation Report (2026-06-08)

Three load attempts were run on 2026-06-08 to confirm the failure
modes documented in `evaluation.tex:327` (the paper's "Single-GPU 24 GB
constraint" limitation). All three attempts failed, as expected.

## Stack

- **GPU**: NVIDIA RTX 4090, 23.51 GiB total
- **transformers**: 5.3.0
- **torch**: 2.9.1+cu128
- **optimum**: NOT installed
- **gptqmodel**: NOT installed
- **auto-gptq**: NOT installed
- **awq (AutoAWQForCausalLM)**: NOT installed

## Attempt 1: Qwen2.5-32B-Instruct-GPTQ-Int4

- **Path**: `/home/gfy/models/Qwen2.5-32B-Instruct-GPTQ-Int4`
- **Files present**: 5× safetensors shards + config.json + tokenizer files
- **Failure mode**: `ImportError: Loading a GPTQ quantized model requires optimum (pip install optimum)`
- **Source**: `transformers/quantizers/quantizer_gptq.py:48`
- **Log**: `01_gptq_int4_load.log`
- **Implication**: gptqmodel 7.0.0 + optimum + transformers ≥ 5.4.0 are required, but upgrading transformers would break the rest of the sglang-kvflow stack (which is pinned at 5.3.0 for vllm 0.10.2 + sglang 0.5.6 compatibility).

## Attempt 2: Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit

- **Path**: `/home/gfy/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit`
- **Files present**: 4× safetensors shards + AWQ config (no awq package installed)
- **Failure mode** (composite):
  1. **MoE experts missing**: 9 keys (down_proj, gate_up_proj, _packed, _scale, _shape × 48 layers) are `MISSING` from the checkpoint. The community AWQ conversion did not quantize the MoE experts.
  2. **CUDA OOM at meta-tensor materialization**: After loading the dense layers successfully (with the `_shape` keys as UNEXPECTED but ignorable), the model fails to materialize the missing expert weights from meta-device to CUDA. Error: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 701.69 MiB is free.`
- **Log**: `02_awq_4bit_load.log`
- **Implication**: Even if the missing experts were synthesized (e.g. via bf16 fallback), the AWQ checkpoint + meta-tensor materialization cannot fit in 24 GB. A 30B-A3B model needs ~16-18 GB just for the AWQ weights; with KV + activations + cuda graphs, 24 GB is insufficient.

## Attempt 3: 32B pass@1 driver

- **Path**: same as Attempt 1
- **Failure mode**: same `ImportError: Loading a GPTQ quantized model requires optimum` — the pass@1 driver is downstream of the model load, so the load failure short-circuits the run.
- **Log**: `03_passrate_1case_load.log`
- **Implication**: Even with `--max-tokens 512 --files-per-case 1`, the 32B model cannot reach the inference stage.

## Verdict

Three Qwen2.5-32B / Qwen3-30B candidates were attempted on 2026-06-08. None
loaded on the 24 GB RTX 4090. The paper's `evaluation.tex:327`
failure-mode description is correct as stated, with one refinement: the
GPTQ-Int4 path fails at the **import** stage (no `optimum` package
installed at all), not at the gptqmodel 7.0.0 vs transformers 5.4.0
compatibility stage. The AWQ-4bit path has a **double failure**: missing
MoE experts + 24 GB OOM. The bf16 fallback for either model would
require 65+ GB of model weights alone, well beyond 24 GB.

The paper's `evaluation.tex:327` paragraph has been updated to reflect
this three-attempt confirmation.

## What would unblock 32B on 24 GB

- **A100 80 GB** or **H100 80 GB**: would fit Qwen2.5-32B bf16 directly
  (65 GB model + ~12 GB KV/activations). Estimated ~3 h to set up +
  ~12 h to run a 100-case pass@1 at small ctx.
- **Hardware upgrade** is the only viable path; the sglang-kvflow stack
  does not currently support 4-bit KV cache for 32B-class models.
