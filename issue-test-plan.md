# Issue #10492 Test Plan

## Summary

This test plan validates the `Scheduler` import-time fix in layers, starting from narrow unit tests and ending with broader runtime and performance checks.

All tests must run inside the active dev container. The performance phase is gated on correctness: no before/after benchmark work starts until the correctness layers are complete.

This plan is synchronized with `issue-test-progress.md`.

## Execution Rules

- Use the active dev container as the only test environment.
- Use the fix branch `issue-10492-scheduler-import-cleanup` for correctness validation.
- After each milestone, stop, summarize the result, and wait for approval before moving to the next milestone.
- Model artifacts must live in a host-mounted cache, not the container's private writable layer.
  - Preferred setup:
    - host: `/home/chris/.cache/huggingface`
    - container: `/root/.cache/huggingface`
  - If a model must be fetched, it may be fetched from inside the container only after the host cache directory is mounted, so the files land on the host-backed cache.
- Treat Llama-based tests as conditional:
  - If `meta-llama/Llama-3.2-1B-Instruct` is accessible in the container, include the Llama path.
  - If access is blocked, continue with the Qwen/TinyLlama/SmolLM/OPT path and record the limitation.

## Test Layers

### 0. Preflight and Model Access

- Confirm branch availability for later before/after comparison:
  - `main`
  - `issue-10492-scheduler-import-cleanup`
- Confirm that the dev container exposes the host Hugging Face cache mount.
- Probe availability of the small-model matrix:
  - `Qwen/Qwen3-0.6B`
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - `HuggingFaceTB/SmolLM-135M-Instruct`
  - `facebook/opt-125m`
  - `meta-llama/Llama-3.2-1B-Instruct`
- Freeze the final correctness matrix based on what is actually accessible from the container.
- Re-bootstrap the Python test environment in the rebuilt container before Layer 1 starts.
  - Success criteria for this bootstrap:
    - a task-local venv exists inside the container
    - `Scheduler`, `ModelConfig`, and `ReasoningParser` all import successfully from the mounted repo

### 1. Narrow Unit Tests

Goal: validate the directly touched import/parser/load paths without starting a server.

- `PYTHONPATH=python python3 -m unittest discover -s test/unit -v`
- `PYTHONPATH=python python3 test/registered/parser/test_reasoning_parser.py`
- `PYTHONPATH=python python3 test/registered/model_loading/test_modelopt_loader.py`
  - If the direct invocation is blocked by the current `sglang.srt.entrypoints` namespace-package patch target, pre-import `sglang.srt.entrypoints.engine` in the same Python process before executing the file and record the workaround.

Expected outcome:

- All import-regression tests pass.
- `reasoning_parser` changes do not break parser behavior.
- loader-side lazy import changes do not break mocked loading flows.

### 2. Broad Unit and Engine Tests

Goal: validate model loading and engine/runtime paths with the lightest practical models.

- Generation parity with tiny models:
  - `ONLY_RUN=HuggingFaceTB/SmolLM-135M-Instruct PYTHONPATH=python python3 test/registered/models/test_generation_models.py`
  - `ONLY_RUN=facebook/opt-125m PYTHONPATH=python python3 test/registered/models/test_generation_models.py`
  - `ONLY_RUN=Qwen/Qwen2-1.5B PYTHONPATH=python python3 test/registered/models/test_generation_models.py`
  - On the local `RTX 2080 SUPER / SM75` machine, treat RMSNorm-heavy CUDA paths as a hardware gate:
    - if a tiny-model parity run still fails after forcing `attention_backend=triton`, `disable_cuda_graph=True`, and `disable_piecewise_cuda_graph=True`, record it as a local kernel-coverage block rather than a regression from the import fix
    - continue the layer with an alternative tiny model such as `facebook/opt-125m`
  - For non-RMSNorm tiny models such as `facebook/opt-125m`, if the direct command is blocked by pre-existing `piecewise_cuda_graph` initialization assumptions, rerun once with `disable_piecewise_cuda_graph=True` and record the fallback.
  - Treat `Qwen/Qwen2-1.5B` as conditional on the local 8 GB GPU budget:
    - if the Hugging Face reference side of the parity test OOMs before SRT comparison starts, record the case as a local-memory block and move on
    - if a reduced local fallback with shorter prompts and fewer generated tokens clears the HF OOM but SRT still fails in `sgl_kernel.rmsnorm`, record the final status as a local SM75 kernel-coverage block
- Engine-level smoke:
  - `cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_engine.TestSRTEngine.test_1_engine_runtime_consistency`
  - `cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_engine.TestSRTEngine.test_3_engine_token_ids_consistency`

Llama conditional path:

- If the Llama model is accessible, the engine smoke above uses that path naturally through `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`.
- If Llama access is blocked, replace this layer's engine/runtime smoke with a public tiny-model fallback that can run locally.
  - On the local 8 GB GPU, `facebook/opt-125m` is the preferred fallback for these engine/runtime checks.
  - If the HTTP `Runtime` wrapper path is blocked by a missing FastAPI multipart dependency, install `python-multipart` in the task-local test environment and rerun once.

### 3. Narrow Integration Smoke

Goal: start real servers with small models and validate the serving path most relevant to the touched scheduler/import code.

Qwen 0.6B path:

- `PYTHONPATH=python python3 test/registered/utils/test_request_logger.py`
- `PYTHONPATH=python python3 test/registered/utils/test_scheduler_status_logger.py`
- `cd test/registered/metrics && PYTHONPATH=../../../python python3 -m unittest -v test_metrics.TestEnableMetrics.test_metrics_1gpu`
  - On the local `RTX 2080 SUPER / SM75` machine, if the real server fails during model init or CUDA graph capture in `sgl_kernel.rmsnorm`, record the entire Qwen runtime smoke path as a local hardware/kernel block rather than a regression from the import-fix changes.
  - If that local Qwen path is blocked, rerun an equivalent model-agnostic server smoke with `facebook/opt-125m` and:
    - `--disable-cuda-graph`
    - `--disable-piecewise-cuda-graph`
    - `--attention-backend triton`
    - `--sampling-backend pytorch`
  - Treat the `facebook/opt-125m` fallback as evidence for request logger, scheduler status, and metrics plumbing only; it does not replace the recorded Qwen-specific SM75 hardware limitation.

Llama conditional path:

- `cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_endpoint.TestSRTEndpoint.test_simple_decode`
- `cd test/registered/openai_server/basic && PYTHONPATH=../../../../python python3 -m unittest -v test_openai_server.TestOpenAIServer.test_completion`
  - On the current container, treat a Hugging Face `401` / gated-repo failure for `meta-llama/Llama-3.2-1B-Instruct` as a local access limitation and record the Llama branch as `skipped`, not as a regression from the import-fix changes.

Expected outcome:

- The server boots successfully with small models.
- Basic generate/OpenAI paths still work.
- request logging, scheduler logging, and metrics remain intact.

### 4. Broader Integration Smoke

Goal: run a small number of broader runtime tests that still fit the local 8 GB GPU.

- `PYTHONPATH=python python3 test/registered/bench_fn/test_bench_serving_functionality.py`
- `PYTHONPATH=python python3 test/registered/scheduler/test_routing_key_scheduling.py`
  - If `test_bench_serving_functionality.py` is blocked locally by the Qwen `SM75` RMSNorm path, keep the file logic but rerun the multi-turn benchmark case with:
    - `facebook/opt-125m`
    - `--disable-cuda-graph`
    - `--disable-piecewise-cuda-graph`
    - `--attention-backend triton`
    - `--sampling-backend pytorch`
    - `--chat-template vicuna_v1.1`
  - The `vicuna_v1.1` chat template is required for the OPT fallback because the benchmark uses the OpenAI chat endpoint and `facebook/opt-125m` does not ship a usable Hugging Face chat template.
  - If `test_routing_key_scheduling.py` is blocked locally by the same Qwen `SM75` RMSNorm path, keep the scheduler-order assertion logic but rerun the case with:
    - `facebook/opt-125m`
    - `--disable-cuda-graph`
    - `--disable-piecewise-cuda-graph`
    - `--attention-backend triton`
    - `--sampling-backend pytorch`
    - `--chat-template vicuna_v1.1`
  - For the OPT fallback only, cap the synthetic long-running requests at a locally valid value such as `max_tokens=512` so the OpenAI chat endpoint stays within the model context budget and does not return request-validation `400`s unrelated to scheduler behavior.

Expected outcome:

- Broader serving/scheduling behavior still works after the import cleanup.
- No obvious runtime regressions are introduced outside the narrow import path.

## Deferred Performance Phase

This phase starts only after Layers 0-4 are green.

### Scope

Measure before/after import performance on both:

- `main`
- `issue-10492-scheduler-import-cleanup`

### Benchmark Matrix

At minimum:

- `import sglang`
- `from sglang.srt.configs.model_config import ModelConfig`
- `from sglang.srt.parser.reasoning_parser import ReasoningParser`
- `from sglang.srt.layers.moe.utils import initialize_moe_config`
- `from sglang.srt.entrypoints.engine import Engine`
- `from sglang.srt.managers.scheduler import Scheduler`

### Method

- Use fresh Python processes for every sample.
- Run 1 warm-up per benchmark item, then 10-20 measured runs.
- Record:
  - `real`
  - `user`
  - `sys`
- Produce:
  - `min`
  - `max`
  - `mean`
  - `median`
- Keep raw samples so the distribution can be inspected if variance is high.

### Comparison Notes

- Before/after comparison requires a clean baseline for both branches.
- The current plan uses:
  - the live fix tree at `/sgl-workspace/sglang`
  - a clean `main` snapshot archived under `/sgl-workspace/sglang/.codex-tmp/main-baseline-20260308-173917`
  - the shared harness `/sgl-workspace/sglang/.codex-tmp/issue10492_import_bench.py`
- This avoids switching the active working tree during measurement and keeps both before/after baselines visible inside the same container.
- Performance conclusions should be based on repeated measurements, not single-run numbers.
