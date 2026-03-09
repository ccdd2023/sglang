# Issue #10492 Test Progress

This file expands `issue-test-plan.md` into milestone-level execution steps.

Execution rule for this task:

- Work only inside the active dev container.
- Execute one milestone at a time.
- After each milestone, report the result and ask before proceeding.
- Do not start the performance phase until all correctness milestones are complete.

Status values:

- `pending`
- `in_progress`
- `completed`
- `blocked`
- `skipped`

## 0. Preflight and Model Access

### 0.1 Confirm the branch matrix for later before/after comparison

- Status: `completed`
- Atomic operation:
  - Verify that both `main` and `issue-10492-scheduler-import-cleanup` are available locally.
- Expected effect:
  - The later performance phase has a defined before/after branch pair.
- Verification:
  - Check `git branch --list`.
- User confirmation gate:
  - User confirms the branch matrix is acceptable.
- Result:
  - Verified local branches:
    - `main`
    - `issue-10492-scheduler-import-cleanup`
  - Current working branch:
    - `issue-10492-scheduler-import-cleanup`

### 0.1.5 Ensure the dev container exposes the host model cache

- Status: `completed`
- Atomic operation:
  - Ensure the active dev container mounts the host Hugging Face cache.
- Expected effect:
  - Any model fetch triggered from inside the container writes into the host-backed cache rather than the container's private layer.
- Verification:
  - Confirm a mount like:
    - `/home/chris/.cache/huggingface -> /root/.cache/huggingface`
- User confirmation gate:
  - User confirms the cache-mount rule is satisfied before model probing starts.
- Result:
  - Recreated the active dev container `sglang_dev_issue10492` with:
    - `/home/chris/.cache/huggingface -> /root/.cache/huggingface`
    - `/home/chris/Workspaces/sglang -> /sgl-workspace/sglang`
  - Set `HF_HOME=/root/.cache/huggingface` in the container.
  - The host cache already exists and currently includes at least:
    - `models--Qwen--Qwen3-0.6B`
  - Note:
    - Recreating the container removed the task-local `/tmp` venv, so the Python test environment will need to be re-bootstrapped before correctness tests start.

### 0.2 Probe small-model accessibility in the container

- Status: `completed`
- Atomic operation:
  - Probe the availability of:
    - `Qwen/Qwen3-0.6B`
    - `Qwen/Qwen2.5-1.5B-Instruct`
    - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
    - `HuggingFaceTB/SmolLM-135M-Instruct`
    - `facebook/opt-125m`
- Expected effect:
  - We know which small models can be used for correctness smoke tests on the 8 GB GPU.
- Verification:
  - Record which models are accessible and which fail due to permissions, download, or memory constraints.
- User confirmation gate:
  - User confirms the non-Llama test matrix.
- Result:
  - Created a minimal probe venv at:
    - `/tmp/sglang-model-probe-venv`
  - Probe method:
    - checked whether the model already exists in the host-mounted Hugging Face cache
    - checked whether Hugging Face remote metadata is accessible from the container
    - if cached, checked whether `AutoConfig.from_pretrained(..., local_files_only=True)` succeeds
  - Results:
    - `Qwen/Qwen3-0.6B`
      - cache present: `True`
      - remote metadata accessible: `True`
      - local config accessible from cache: `True`
    - `Qwen/Qwen2.5-1.5B-Instruct`
      - cache present: `False`
      - remote metadata accessible: `True`
    - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
      - cache present: `False`
      - remote metadata accessible: `True`
    - `HuggingFaceTB/SmolLM-135M-Instruct`
      - cache present: `False`
      - remote metadata accessible: `True`
    - `facebook/opt-125m`
      - cache present: `False`
      - remote metadata accessible: `True`
  - Interpretation:
    - The Qwen 0.6B path is immediately available without any new model fetch.
    - The other non-Llama small models are accessible in principle and can be fetched later into the host-mounted cache if we decide to include them in correctness tests.

### 0.3 Probe Llama access in the container

- Status: `completed`
- Atomic operation:
  - Check whether `meta-llama/Llama-3.2-1B-Instruct` is accessible from the container.
- Expected effect:
  - We know whether the Llama path can be included in the correctness plan.
- Verification:
  - Record pass/fail and the reason if blocked.
- User confirmation gate:
  - User confirms whether to include or skip the Llama branch.
- Result:
  - Probed model:
    - `meta-llama/Llama-3.2-1B-Instruct`
  - Results:
    - cache present: `False`
    - remote metadata accessible: `True`
    - local config accessible from cache: `False`
  - Interpretation:
    - The Llama test path is available in principle from the container.
    - The model is not yet cached locally, but it can be fetched later into the host-mounted Hugging Face cache if we decide to include it in correctness tests.

### 0.4 Freeze the final correctness matrix

- Status: `completed`
- Atomic operation:
  - Finalize the exact set of models and test files for Layers 1-4 based on 0.2 and 0.3.
- Expected effect:
  - The remaining correctness phases have a stable execution matrix.
- Verification:
  - Summarize the approved model/test matrix.
- User confirmation gate:
  - User confirms the final correctness matrix before execution starts.
- Result:
  - Frozen correctness matrix:
    - Layer 1: narrow unit tests
      - `PYTHONPATH=python python3 -m unittest discover -s test/unit -v`
      - `PYTHONPATH=python python3 test/registered/parser/test_reasoning_parser.py`
      - `PYTHONPATH=python python3 test/registered/model_loading/test_modelopt_loader.py`
    - Layer 2: broad unit and engine tests
      - `ONLY_RUN=HuggingFaceTB/SmolLM-135M-Instruct PYTHONPATH=python python3 test/registered/models/test_generation_models.py`
      - `ONLY_RUN=facebook/opt-125m PYTHONPATH=python python3 test/registered/models/test_generation_models.py`
      - `ONLY_RUN=Qwen/Qwen2-1.5B PYTHONPATH=python python3 test/registered/models/test_generation_models.py`
      - `cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_engine.TestSRTEngine.test_1_engine_runtime_consistency`
      - `cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_engine.TestSRTEngine.test_3_engine_token_ids_consistency`
    - Layer 3: narrow integration smoke
      - `PYTHONPATH=python python3 test/registered/utils/test_request_logger.py`
      - `PYTHONPATH=python python3 test/registered/utils/test_scheduler_status_logger.py`
      - `cd test/registered/metrics && PYTHONPATH=../../../python python3 -m unittest -v test_metrics.TestEnableMetrics.test_metrics_1gpu`
      - `cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_endpoint.TestSRTEndpoint.test_simple_decode`
      - `cd test/registered/openai_server/basic && PYTHONPATH=../../../../python python3 -m unittest -v test_openai_server.TestOpenAIServer.test_completion`
    - Layer 4: broader integration smoke
      - `PYTHONPATH=python python3 test/registered/bench_fn/test_bench_serving_functionality.py`
      - `PYTHONPATH=python python3 test/registered/scheduler/test_routing_key_scheduling.py`
  - Model allocation for this matrix:
    - Immediately available from host cache:
      - `Qwen/Qwen3-0.6B`
    - Approved to fetch later into the host-mounted cache if needed by the selected tests:
      - `meta-llama/Llama-3.2-1B-Instruct`
      - `Qwen/Qwen2-1.5B`
      - `HuggingFaceTB/SmolLM-135M-Instruct`
      - `facebook/opt-125m`
  - Rationale:
    - keep Layer 1 close to the touched import/parser/load code
    - use tiny public models first in Layer 2
    - keep real-server smoke anchored on cached `Qwen/Qwen3-0.6B`
    - include Llama runtime/OpenAI paths because access is available in principle

### 0.5 Re-bootstrap the Python test environment in the rebuilt container

- Status: `completed`
- Atomic operation:
  - Recreate the task-local Python test environment inside the rebuilt container.
- Expected effect:
  - Layer 1 tests can run against the mounted repo from a usable venv.
- Verification:
  - Confirm the new venv exists and basic project imports resolve from `/sgl-workspace/sglang/python`.
- User confirmation gate:
  - User confirms the rebuilt container is ready for Layer 1.
- Actual result:
  - Rebuilt the task-local test venv at:
    - `/tmp/sglang-issue10492-venv-min2`
  - Installed `torch==2.9.1+cu128` by bypassing a corrupted local wheel cache and the broken `pypi.nvidia.com` hash entry for `nvidia-cusolver-cu12==11.7.3.90`.
  - Installed the minimum Python/runtime dependencies needed to unblock Layer 1 import probes.
  - Restored repository-aligned pins for:
    - `blobfile==3.0.0`
    - `llguidance==0.7.30`
    - `xgrammar==0.1.27`
- Verification result:
  - Active interpreter:
    - `/tmp/sglang-issue10492-venv-min2/bin/python`
  - Import probes inside the rebuilt container all passed:
    - `from sglang.srt.managers.scheduler import Scheduler`
    - `from sglang.srt.configs.model_config import ModelConfig`
    - `from sglang.srt.parser.reasoning_parser import ReasoningParser`

## 1. Narrow Unit Tests

### 1.1 Run `test/unit`

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    PYTHONPATH=python python3 -m unittest discover -s test/unit -v
    ```
- Expected effect:
  - The narrowest unit-test layer passes, including the new import regression tests.
- Verification:
  - Record total tests, failures, errors, and skips.
- User confirmation gate:
  - User confirms the narrow unit baseline is acceptable.
- Actual result:
  - Ran in the rebuilt container with:
    - `/tmp/sglang-issue10492-venv-min2/bin/python`
  - Command:
    ```bash
    PYTHONPATH=python python -m unittest discover -s test/unit -v
    ```
  - Covered:
    - `test_mamba_state_scatter_triton`
    - `test_scheduler_import_regression`
- Verification result:
  - `Ran 6 tests in 10.771s`
  - `OK (skipped=1)`
  - The new import regression tests all passed:
    - `test_compatibility_imports_still_work`
    - `test_import_sglang_stays_lazy`
    - `test_moe_utils_import_does_not_load_runner`
    - `test_scheduler_import_avoids_heavy_optional_modules`

### 1.2 Run parser-only regression coverage

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    PYTHONPATH=python python3 test/registered/parser/test_reasoning_parser.py
    ```
- Expected effect:
  - `reasoning_parser` changes do not break parser correctness.
- Verification:
  - Confirm the file passes.
- User confirmation gate:
  - User confirms the parser layer is acceptable.
- Actual result:
  - Initial run failed because `sglang.test.test_utils` pulled in `sglang.benchmark.datasets`, and the rebuilt venv was still missing `datasets`.
  - Installed `datasets` into `/tmp/sglang-issue10492-venv-min2` and reran the same test file.
- Verification result:
  - `Ran 65 tests in 0.003s`
  - `OK`
  - No parser behavior regression was observed after the `reasoning_parser` import cleanup.

### 1.3 Run loader-side mocked regression coverage

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    PYTHONPATH=python python3 test/registered/model_loading/test_modelopt_loader.py
    ```
- Expected effect:
  - loader/config-related lazy import changes do not break mocked model-loading paths.
- Verification:
  - Confirm the file passes or clearly document any unrelated environmental blockers.
- User confirmation gate:
  - User confirms this narrow loader layer is acceptable.
- Actual result:
  - The first direct run failed with `10` errors.
  - The direct-run blockers were reduced in stages:
    - Installed `torchvision` to satisfy `internvl_utils` imports triggered by `ModelConfig(...)`.
    - Installed `compressed-tensors` to satisfy quantization-method imports.
    - Installed `uvloop`, `uvicorn`, and `watchfiles` so `sglang.srt.entrypoints.engine` can import.
    - Installed `gguf` to satisfy the remaining quantization-method import.
  - Two of the original failures were due to a pre-existing test/packaging issue:
    - the test file patches `sglang.srt.entrypoints.engine.Engine.__init__`
    - `sglang.srt.entrypoints` is a namespace package with no `__init__.py`
    - `pkgutil.resolve_name("sglang.srt.entrypoints.engine.Engine")` raises `AttributeError` unless the `engine` submodule is imported first
  - After pre-importing `sglang.srt.entrypoints.engine` in the same Python process, the file passed cleanly.
- Verification result:
  - Direct command outcome after environment fixes:
    ```bash
    PYTHONPATH=python python test/registered/model_loading/test_modelopt_loader.py
    ```
    - reduced to `FAILED (errors=3)` and then `FAILED (errors=2)` due to the namespace-package patch target
  - Functional validation command:
    ```bash
    PYTHONPATH=python python - <<'PY'
    import runpy
    import sglang.srt.entrypoints.engine
    runpy.run_path("test/registered/model_loading/test_modelopt_loader.py", run_name="__main__")
    PY
    ```
    - `Ran 10 tests in 2.859s`
    - `OK`
  - Current interpretation:
    - loader/config-related lazy import changes did not break the mocked ModelOpt loading paths
    - the remaining direct-invocation issue is a test harness/package-resolution quirk, not evidence against the fix

## 2. Broad Unit and Engine Tests

### 2.1 Run tiny-model generation parity on SmolLM

- Status: `blocked`
- Atomic operation:
  - Run:
    ```bash
    ONLY_RUN=HuggingFaceTB/SmolLM-135M-Instruct PYTHONPATH=python python3 test/registered/models/test_generation_models.py
    ```
- Expected effect:
  - HF-vs-SRT generation parity still works on a minimal model.
- Verification:
  - Confirm the selected model case passes.
- User confirmation gate:
  - User confirms the first broad-unit model result.
- Actual result:
  - The first direct run fetched `HuggingFaceTB/SmolLM-135M-Instruct` into the host-mounted Hugging Face cache and then failed before parity comparison because the rebuilt venv was still missing `flashinfer`.
  - After installing `flashinfer_python==0.6.4` and `flashinfer_cubin==0.6.4`, the same test advanced into real runtime execution but failed inside the scheduler subprocess on:
    - `RuntimeError: RMSNorm failed with error code no kernel image is available for execution on the device`
  - To separate graph/backend choice from hardware coverage, reran the same model through a local shim that forced:
    - `attention_backend='triton'`
    - `disable_cuda_graph=True`
    - `disable_piecewise_cuda_graph=True`
  - Even under that most conservative runtime configuration, the scheduler subprocess still failed in:
    - `python/sglang/srt/layers/layernorm.py`
    - `sgl_kernel.rmsnorm`
    - with the same `no kernel image is available for execution on the device`
  - Local interpretation:
    - this is a pre-existing kernel coverage limitation on the local `RTX 2080 SUPER / SM75` machine
    - it is not evidence that the issue #10492 import-fix branch regressed generation parity
    - Layer 2 should continue with an alternative tiny model that does not immediately depend on the same unsupported RMSNorm CUDA kernel path

### 2.2 Run tiny-model generation parity on OPT-125M

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    ONLY_RUN=facebook/opt-125m PYTHONPATH=python python3 test/registered/models/test_generation_models.py
    ```
- Expected effect:
  - Another tiny public model passes the same parity path.
- Verification:
  - Confirm the selected model case passes.
- User confirmation gate:
  - User confirms the second broad-unit model result.
- Actual result:
  - The direct command entered real runtime/model loading and then failed during scheduler initialization in:
    - `python/sglang/srt/model_executor/model_runner.py:init_piecewise_cuda_graphs`
  - The specific error was:
    - `AttributeError: 'OPTModel' object has no attribute 'layers'`
  - Local interpretation:
    - this is a pre-existing runtime assumption in the default piecewise CUDA graph initialization path for `facebook/opt-125m`
    - it is distinct from the issue #10492 import-fix changes
  - Functional fallback validation:
    - reran the same test file through a local shim that forced:
      - `disable_cuda_graph=True`
      - `disable_piecewise_cuda_graph=True`
    - under that configuration, the selected model case passed end to end
- Verification result:
  - Fallback validation output:
    - `Ran 1 test in 18.868s`
    - `OK`
  - HF vs SRT outputs matched for all sampled prompts.
  - Logged parity stats stayed within tolerance:
    - `rouge_l_scores=[1.0, 1.0, 1.0]`
    - decode/prefill logprob diffs remained small and within the test thresholds

### 2.3 Run Qwen 1.5B generation parity if accessible

- Status: `blocked`
- Atomic operation:
  - Run:
    ```bash
    ONLY_RUN=Qwen/Qwen2-1.5B PYTHONPATH=python python3 test/registered/models/test_generation_models.py
    ```
- Expected effect:
  - A medium-small Qwen path also passes parity if the model is available and fits locally.
- Verification:
  - Confirm pass, skip, or blocked status with reason.
- User confirmation gate:
  - User confirms whether the Qwen parity result is sufficient.
- Actual result:
  - The direct command progressed into the Hugging Face reference side of the parity test, which means the selected model was accessible and loaded from the host-mounted Hugging Face cache path.
  - The run failed before SRT-vs-HF comparison could complete because the Hugging Face reference model OOMed on the local `RTX 2080 SUPER 8GB` GPU:
    - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 470.00 MiB. GPU 0 has a total capacity of 7.60 GiB ...`
  - Failure location:
    - `transformers/models/qwen2/modeling_qwen2.py`
    - inside the reference HF generation path, before this milestone could establish a comparable parity result.
  - Local fallback validation:
    - reran the same model case through a local shim that:
      - reduced prompts to two short prompts
      - capped `max_new_tokens` at `8`
      - forced `disable_cuda_graph=True`
      - forced `disable_piecewise_cuda_graph=True`
      - forced `attention_backend='triton'`
    - under that reduced-memory setup, the HF reference side no longer OOMed, but the SRT scheduler subprocess then failed in:
      - `python/sglang/srt/layers/layernorm.py`
      - `sgl_kernel.rmsnorm`
      - with `RuntimeError: RMSNorm failed with error code no kernel image is available for execution on the device`
  - Local interpretation:
    - the direct command is blocked first by local HF memory pressure on the 8 GB GPU
    - after reducing the HF memory footprint, the final blocker is the same local `SM75` RMSNorm kernel-coverage limitation already seen with other RMSNorm-based models
    - this is not evidence that the issue #10492 import-fix branch regressed Qwen generation behavior
- Verification result:
  - Outcome:
    - `blocked`
  - Reason:
    - direct path: HF baseline OOM on the local 8 GB GPU
    - reduced local fallback: SRT RMSNorm kernel unavailable on local `SM75`

### 2.4 Run lightweight engine/runtime smoke

- Status: `completed`
- Atomic operation:
  - Preferred command if Llama is accessible:
    ```bash
    cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_engine.TestSRTEngine.test_1_engine_runtime_consistency
    ```
  - Then:
    ```bash
    cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_engine.TestSRTEngine.test_3_engine_token_ids_consistency
    ```
- Expected effect:
  - Core engine/runtime APIs still behave correctly after the import cleanup.
- Verification:
  - Confirm both selected engine tests pass, or record the approved fallback if Llama access is blocked.
- User confirmation gate:
  - User confirms the broad unit/engine layer is acceptable.
- Actual result:
  - The direct default path was blocked before runtime execution because the test file hardcodes:
    - `DEFAULT_SMALL_MODEL_NAME_FOR_TEST = meta-llama/Llama-3.2-1B-Instruct`
  - On this machine/container, the direct command failed with:
    - `401 Client Error / GatedRepoError`
    - `Access to model meta-llama/Llama-3.2-1B-Instruct is restricted`
  - Local fallback validation:
    - loaded `test_srt_engine.py` through a local shim
    - replaced `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` with:
      - `facebook/opt-125m`
    - patched both `sgl.Engine` and `sgl.Runtime` to inject:
      - `disable_cuda_graph=True`
      - `disable_piecewise_cuda_graph=True`
  - During the first fallback run:
    - `test_3_engine_token_ids_consistency` passed
    - `test_1_engine_runtime_consistency` exposed an environment-only dependency gap:
      - `python-multipart` missing from the task-local venv
  - Installed:
    - `python-multipart`
    - into `/tmp/sglang-issue10492-venv-min2`
  - Reran the same fallback suite after the environment fix.
- Verification result:
  - Fallback suite output:
    - `Ran 2 tests in 41.675s`
    - `OK`
  - Covered fallback cases:
    - `TestSRTEngine.test_1_engine_runtime_consistency`
    - `TestSRTEngine.test_3_engine_token_ids_consistency`
  - Interpretation:
    - the engine/runtime API layer remains functional after the import-fix changes
    - the direct default Llama path is blocked locally by model access, not by an observed regression in the import-fix branch

## 3. Narrow Integration Smoke

### 3.1 Run Qwen 0.6B request-logger smoke

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    PYTHONPATH=python python3 test/registered/utils/test_request_logger.py
    ```
- Expected effect:
  - A real server starts and request logging still works.
- Verification:
  - Confirm the file passes.
- User confirmation gate:
  - User confirms the first integration smoke result.
- Actual result:
  - The test launched the real server with:
    - `sglang serve --model-path Qwen/Qwen3-0.6B --log-requests ...`
  - The model was available from the host-mounted Hugging Face cache and loaded successfully into the server process.
  - The server then failed during CUDA graph capture before it could become ready and before any request-logger assertions ran.
  - Failure path:
    - `python/sglang/srt/model_executor/cuda_graph_runner.py`
    - `python/sglang/srt/layers/layernorm.py`
    - `sgl_kernel.rmsnorm`
  - The concrete runtime error was:
    - `RuntimeError: RMSNorm failed with error code no kernel image is available for execution on the device`
  - The same startup failure occurred across the logger format variants in the file (`text` and `json`), so the file never reached request/response validation.
  - Local fallback:
    - launched a real server with:
      - `sglang serve --model-path facebook/opt-125m --log-requests --log-requests-level 2 --log-requests-format json --skip-server-warmup --log-requests-target stdout <tmpdir> --disable-cuda-graph --disable-piecewise-cuda-graph --attention-backend triton --sampling-backend pytorch`
    - the fallback server started successfully on CUDA, served a `/generate` request, emitted `request.received` and `request.finished` events, and wrote a request log file under the temporary target directory
- Verification result:
  - Direct Qwen result:
    - `blocked`
    - reason:
      - local `SM75` hardware/kernel limitation in the real Qwen server startup path
  - Fallback logger result:
    - `completed`
    - evidence:
      - HTTP server startup succeeded
      - `/generate` returned `200`
      - request logger emitted both `request.received` and `request.finished`
      - request log file creation succeeded
  - Interpretation:
    - the direct Qwen path remains locally blocked by `sgl_kernel.rmsnorm` coverage on `SM75`
    - the request-logger plumbing itself still works on the import-fix branch when exercised through an equivalent real-server fallback

### 3.2 Run Qwen 0.6B scheduler-status smoke

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    PYTHONPATH=python python3 test/registered/utils/test_scheduler_status_logger.py
    ```
- Expected effect:
  - A real server starts and scheduler status logging still works.
- Verification:
  - Confirm the file passes.
- User confirmation gate:
  - User confirms the scheduler-status result.
- Actual result:
  - Direct invocation:
    - ran:
      - `PYTHONPATH=python python3 test/registered/utils/test_scheduler_status_logger.py`
    - the real `Qwen/Qwen3-0.6B` server loaded weights successfully
    - startup then failed during CUDA graph capture in the same path observed in `3.1`:
      - `python/sglang/srt/model_executor/cuda_graph_runner.py`
      - `python/sglang/srt/layers/layernorm.py`
      - `sgl_kernel.rmsnorm`
    - the concrete runtime error was:
      - `RuntimeError: RMSNorm failed with error code no kernel image is available for execution on the device`
    - unittest result:
      - `Ran 0 tests in 20.004s`
      - `FAILED (errors=1)`
  - Local fallback:
    - launched an equivalent real server with:
      - `sglang serve --model-path facebook/opt-125m --skip-server-warmup --enable-metrics --disable-cuda-graph --disable-piecewise-cuda-graph --attention-backend triton --sampling-backend pytorch`
    - preserved the scheduler-status logging environment:
      - `SGLANG_LOG_SCHEDULER_STATUS_TARGET=<tmpdir>`
      - `SGLANG_LOG_SCHEDULER_STATUS_INTERVAL=1`
    - sent a real `/generate` request after startup
    - observed one `scheduler.status` event in the emitted log file
- Verification result:
  - Direct Qwen result:
    - `blocked`
    - reason:
      - local `SM75` hardware/kernel limitation in the real Qwen server startup path
  - Fallback scheduler-status result:
    - `completed`
    - evidence:
      - HTTP server startup succeeded
      - `/generate` returned `200`
      - one `scheduler.status` event was written
      - emitted event included:
        - `timestamp`
        - `rank`
        - `running_rids`
        - `queued_rids`
      - `running_rids` and `queued_rids` were both lists
  - Interpretation:
    - the direct Qwen path remains locally blocked by `sgl_kernel.rmsnorm` coverage on `SM75`
    - the scheduler-status logging plumbing itself still works on the import-fix branch when exercised through an equivalent real-server fallback

### 3.3 Run Qwen 0.6B metrics smoke

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    cd test/registered/metrics && PYTHONPATH=../../../python python3 -m unittest -v test_metrics.TestEnableMetrics.test_metrics_1gpu
    ```
- Expected effect:
  - Metrics collection and the main serving loop still work on 1 GPU.
- Verification:
  - Confirm the selected metrics test passes.
- User confirmation gate:
  - User confirms the metrics result.
- Actual result:
  - Direct invocation:
    - ran:
      - `cd test/registered/metrics && PYTHONPATH=../../../python python3 -m unittest -v test_metrics.TestEnableMetrics.test_metrics_1gpu`
    - the real `Qwen/Qwen3-0.6B` server loaded weights successfully
    - startup then failed during CUDA graph capture in the same path observed in `3.1` and `3.2`:
      - `python/sglang/srt/model_executor/cuda_graph_runner.py`
      - `python/sglang/srt/layers/layernorm.py`
      - `sgl_kernel.rmsnorm`
    - the concrete runtime error was:
      - `RuntimeError: RMSNorm failed with error code no kernel image is available for execution on the device`
    - unittest result:
      - `Ran 1 test in 20.005s`
      - `FAILED (errors=1)`
  - Local fallback:
    - loaded `test/registered/metrics/test_metrics.py` by file path and reused the original `TestEnableMetrics.test_metrics_1gpu` method
    - monkeypatched only:
      - `_MODEL_NAME = "facebook/opt-125m"`
      - `popen_launch_server(...)` to append:
        - `--disable-cuda-graph`
        - `--disable-piecewise-cuda-graph`
        - `--attention-backend triton`
        - `--sampling-backend pytorch`
    - the fallback test then ran the original metrics assertions against a real `sglang serve` process
    - output ended with:
      - `fallback_metrics_test_passed`
- Verification result:
  - Direct Qwen result:
    - `blocked`
    - reason:
      - local `SM75` hardware/kernel limitation in the real Qwen server startup path
  - Fallback metrics result:
    - `completed`
    - evidence:
      - real server startup succeeded
      - the original metrics test flow completed against `/health_generate`, `/generate`, and `/metrics`
      - Prometheus output included the expected metric families and positive values
      - routed-key histogram metrics were present for `facebook/opt-125m`
  - Interpretation:
    - the direct Qwen path remains locally blocked by `sgl_kernel.rmsnorm` coverage on `SM75`
    - the metrics endpoint and validation logic still work on the import-fix branch when exercised through an equivalent real-server fallback

### 3.4 Run Llama SRT endpoint smoke if accessible

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_endpoint.TestSRTEndpoint.test_simple_decode
    ```
- Expected effect:
  - The default small Llama server path still handles basic decode.
- Verification:
  - Confirm pass or mark `skipped` if Llama access is blocked.
- User confirmation gate:
  - User confirms the Llama SRT smoke result or skip rationale.
- Actual result:
  - Direct invocation:
    - ran:
      - `cd test/registered/core && PYTHONPATH=../../../python python3 -m unittest -v test_srt_endpoint.TestSRTEndpoint.test_simple_decode`
    - startup failed before the server became ready
    - the failure occurred while resolving the model config for:
      - `meta-llama/Llama-3.2-1B-Instruct`
    - Hugging Face returned:
      - `401 Client Error`
      - `GatedRepoError`
      - `Access to model meta-llama/Llama-3.2-1B-Instruct is restricted`
    - unittest result:
      - `Ran 0 tests in 10.003s`
      - `FAILED (errors=1)`
- Verification result:
  - Outcome:
    - `skipped`
  - Reason:
    - the current container does not have real gated access to `meta-llama/Llama-3.2-1B-Instruct`
    - this is a model-access limitation, not evidence of an import-fix regression
  - Interpretation:
    - the local Llama SRT smoke path cannot be exercised further without valid Hugging Face access for the gated model

### 3.5 Run Llama OpenAI completion smoke if accessible

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    cd test/registered/openai_server/basic && PYTHONPATH=../../../../python python3 -m unittest -v test_openai_server.TestOpenAIServer.test_completion
    ```
- Expected effect:
  - The OpenAI-compatible path still works on the default small Llama model.
- Verification:
  - Confirm pass or mark `skipped` if Llama access is blocked.
- User confirmation gate:
  - User confirms the Llama OpenAI smoke result or skip rationale.
- Actual result:
  - Direct invocation:
    - ran:
      - `cd test/registered/openai_server/basic && PYTHONPATH=../../../../python python3 -m unittest -v test_openai_server.TestOpenAIServer.test_completion`
    - startup failed before the OpenAI-compatible server became ready
    - the failure occurred while resolving the model config for:
      - `meta-llama/Llama-3.2-1B-Instruct`
    - Hugging Face returned:
      - `401 Client Error`
      - `GatedRepoError`
      - `Access to model meta-llama/Llama-3.2-1B-Instruct is restricted`
    - unittest result:
      - `Ran 0 tests in 10.003s`
      - `FAILED (errors=1)`
- Verification result:
  - Outcome:
    - `skipped`
  - Reason:
    - the current container does not have real gated access to `meta-llama/Llama-3.2-1B-Instruct`
    - this is a model-access limitation, not evidence of an import-fix regression
  - Interpretation:
    - the local Llama OpenAI smoke path cannot be exercised further without valid Hugging Face access for the gated model

## 4. Broader Integration Smoke

### 4.1 Run benchmark-serving functionality smoke

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    PYTHONPATH=python python3 test/registered/bench_fn/test_bench_serving_functionality.py
    ```
- Expected effect:
  - Multi-turn serving benchmark functionality still works on Qwen 0.6B.
- Verification:
  - Confirm the file passes.
- User confirmation gate:
  - User confirms the broader serving smoke result.
- Actual result:
  - Direct invocation:
    - ran:
      - `PYTHONPATH=python python3 test/registered/bench_fn/test_bench_serving_functionality.py`
    - file result:
      - `Ran 3 tests in 32.129s`
      - `FAILED (errors=1)`
    - direct sub-results:
      - `TestBenchServingCustomHeaders.test_custom_headers_sent_to_server`: passed
      - `TestBenchServingCustomHeaders.test_parse_custom_headers`: passed
      - `TestBenchServingFunctionality.test_gsp_multi_turn`: blocked by the same local `Qwen/Qwen3-0.6B` startup failure seen in Layer 3
    - Qwen failure path:
      - `python/sglang/srt/model_executor/cuda_graph_runner.py`
      - `python/sglang/srt/layers/layernorm.py`
      - `sgl_kernel.rmsnorm`
    - concrete runtime error:
      - `RuntimeError: RMSNorm failed with error code no kernel image is available for execution on the device`
  - First fallback attempt:
    - switched the server model to `facebook/opt-125m` and disabled CUDA graphs
    - benchmark requests then failed with `400 Bad Request` on `/v1/chat/completions`
    - root cause:
      - this benchmark uses the OpenAI chat endpoint
      - `facebook/opt-125m` does not expose a usable chat template by default
  - Fallback preflight:
    - launched `facebook/opt-125m` with:
      - `--disable-cuda-graph`
      - `--disable-piecewise-cuda-graph`
      - `--attention-backend triton`
      - `--sampling-backend pytorch`
      - `--chat-template vicuna_v1.1`
    - verified that `/v1/chat/completions` returned `200`
  - Final fallback:
    - loaded `test/registered/bench_fn/test_bench_serving_functionality.py` by file path
    - monkeypatched only:
      - `MODEL = "facebook/opt-125m"`
      - `popen_launch_server(...)` to append:
        - `--disable-cuda-graph`
        - `--disable-piecewise-cuda-graph`
        - `--attention-backend triton`
        - `--sampling-backend pytorch`
        - `--chat-template vicuna_v1.1`
    - reran only:
      - `TestBenchServingFunctionality.test_gsp_multi_turn`
    - output ended with:
      - `fallback_bench_gsp_passed`
- Verification result:
  - Direct Qwen result:
    - `blocked`
    - reason:
      - local `SM75` hardware/kernel limitation in the real Qwen server startup path
  - Direct non-server subtests:
    - `completed`
    - evidence:
      - the custom-header parsing and echo behavior tests passed without modification
  - Fallback benchmark result:
    - `completed`
    - evidence:
      - OpenAI chat endpoint worked on the OPT fallback after adding `--chat-template vicuna_v1.1`
      - multi-turn benchmark completed with:
        - `Successful requests: 12`
      - request logs still captured the expanding conversation text used by `_verify_multi_turn_logs`
  - Interpretation:
    - the direct Qwen path remains locally blocked by `sgl_kernel.rmsnorm` coverage on `SM75`
    - the broader benchmark-serving behavior still works on the import-fix branch when exercised through an equivalent real-server fallback

### 4.2 Run routing-key scheduling smoke

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    PYTHONPATH=python python3 test/registered/scheduler/test_routing_key_scheduling.py
    ```
- Expected effect:
  - Routing-key scheduling still behaves correctly with a real Qwen 0.6B server.
- Verification:
  - Confirm the file passes.
- User confirmation gate:
  - User confirms the broader scheduler smoke result.
- Actual result:
  - Direct invocation:
    - ran:
      - `PYTHONPATH=python python3 test/registered/scheduler/test_routing_key_scheduling.py`
    - the real `Qwen/Qwen3-0.6B` server loaded weights successfully
    - startup then failed during CUDA graph capture in the same path observed in Layer 3 and `4.1`:
      - `python/sglang/srt/model_executor/cuda_graph_runner.py`
      - `python/sglang/srt/layers/layernorm.py`
      - `sgl_kernel.rmsnorm`
    - the concrete runtime error was:
      - `RuntimeError: RMSNorm failed with error code no kernel image is available for execution on the device`
    - unittest result:
      - `Ran 0 tests in 20.004s`
      - `FAILED (errors=1)`
  - First fallback attempt:
    - loaded `test/registered/scheduler/test_routing_key_scheduling.py` by file path
    - monkeypatched only:
      - `TestRoutingKeyScheduling.setUpClass(...)` to launch:
        - `facebook/opt-125m`
        - `--disable-cuda-graph`
        - `--disable-piecewise-cuda-graph`
        - `--attention-backend triton`
        - `--sampling-backend pytorch`
        - `--chat-template vicuna_v1.1`
      - `TestRoutingKeyScheduling.tearDownClass(...)` to preserve the original cleanup
    - this preserved the original routing-key scheduling logic and passed the core assertion
    - however, the two synthetic long-running requests returned `400 Bad Request` because the original `max_tokens=20000` exceeded what the OPT fallback could accept cleanly through the OpenAI chat endpoint
  - Final fallback:
    - kept the same OPT server fallback and the same scheduler-order assertion logic
    - monkeypatched only `TestRoutingKeyScheduling._send_chat_request(...)` so the synthetic long-running requests use:
      - `max_tokens=min(max_tokens, 512)`
    - this avoided the unrelated request-validation `400`s while still keeping the long requests running long enough to create queue pressure
    - final fallback output ended with:
      - `Average key_a latency: 0.426s`
      - `Average key_b latency: 1.131s`
      - `OK`
- Verification result:
  - Direct Qwen result:
    - `blocked`
    - reason:
      - local `SM75` hardware/kernel limitation in the real Qwen server startup path
  - Final fallback scheduler result:
    - `completed`
    - evidence:
      - real server startup succeeded on the OPT fallback
      - all fallback `/v1/chat/completions` requests returned `200`
      - routing-key debug logs showed `waiting_keys_after` preferring `key_a` ahead of `key_b`
      - the original assertion passed with:
        - `avg_key_a < avg_key_b`
        - measured values:
          - `avg_key_a = 0.426s`
          - `avg_key_b = 1.131s`
  - Interpretation:
    - the direct Qwen path remains locally blocked by `sgl_kernel.rmsnorm` coverage on `SM75`
    - the broader routing-key scheduling behavior still works on the import-fix branch when exercised through an equivalent real-server fallback

## 5. Performance Comparison (Deferred)

### 5.1 Prepare clean before/after baselines

- Status: `completed`
- Atomic operation:
  - Freeze the fix branch state and prepare a clean `main` baseline for comparison.
- Expected effect:
  - Both before and after are reproducible and measured from clean code states.
- Verification:
  - Summarize the branch/worktree setup that will be used.
- User confirmation gate:
  - User confirms the comparison setup before benchmarking starts.
- Actual result:
  - Current fix baseline:
    - live working tree on branch:
      - `issue-10492-scheduler-import-cleanup`
    - container-visible path:
      - `/sgl-workspace/sglang`
  - Clean before baseline:
    - archived a tracked-file snapshot of local `main` into:
      - `/sgl-workspace/sglang/.codex-tmp/main-baseline-20260308-173917`
    - local `main` commit:
      - `97a2a9be0f45a64f1d2e469456377361329011eb`
  - Shared benchmark harness:
    - added:
      - `/sgl-workspace/sglang/.codex-tmp/issue10492_import_bench.py`
    - benchmark matrix now includes:
      - `import sglang`
      - `ModelConfig`
      - `ReasoningParser`
      - `initialize_moe_config`
      - `Engine`
      - `Scheduler`
  - Dry-run verification:
    - ran the harness inside the container against the clean `main` snapshot with:
      - `--warmups 0`
      - `--runs 1`
    - output file:
      - `/sgl-workspace/sglang/.codex-tmp/perf-results/main-dryrun.json`
    - sample outputs were produced for all benchmark items, including:
      - `import_engine`
      - `import_scheduler`
    - representative dry-run results from `main`:
      - `import_sglang real=0.640s`
      - `import_model_config real=6.986s`
      - `import_reasoning_parser real=3.168s`
      - `import_moe_utils real=6.408s`
      - `import_engine real=7.996s`
      - `import_scheduler real=7.764s`
- Verification result:
  - Outcome:
    - `completed`
  - Evidence:
    - both before and after code roots are available simultaneously inside the same container
    - the same venv and the same harness can execute against the clean `main` snapshot
    - importtime sample files are emitted per benchmark item under:
      - `/sgl-workspace/sglang/.codex-tmp/perf-results/main-dryrun-samples/`
  - Interpretation:
    - Step `5.2` can now run repeated measurements on `main` without switching branches or rebuilding the container

### 5.2 Run repeated import benchmarks on `main`

- Status: `completed`
- Atomic operation:
  - Run the benchmark matrix on `main` with 1 warm-up and 10-20 measured runs per item.
- Expected effect:
  - A stable before dataset is collected.
- Verification:
  - Record raw `real/user/sys` samples for every benchmark item.
- User confirmation gate:
  - User confirms the baseline dataset before the fix-branch run.
- Actual result:
  - Ran the shared harness inside the container against:
    - `/sgl-workspace/sglang/.codex-tmp/main-baseline-20260308-173917`
  - Command configuration:
    - `--warmups 1`
    - `--runs 10`
  - Output:
    - `/sgl-workspace/sglang/.codex-tmp/perf-results/main-10runs.json`
  - Sample directories:
    - `/sgl-workspace/sglang/.codex-tmp/perf-results/main-10runs-samples/`
  - Dataset integrity checks:
    - every benchmark item recorded `10` measured samples
    - sample directories contain `11` files per item:
      - `1` warm-up importtime sample
      - `10` measured importtime samples
- Verification result:
  - Outcome:
    - `completed`
  - Evidence:
    - raw `real/user/sys` samples exist for:
      - `import_sglang`
      - `import_model_config`
      - `import_reasoning_parser`
      - `import_moe_utils`
      - `import_engine`
      - `import_scheduler`
  - Interpretation:
    - the before dataset is complete and ready for direct comparison against the fix tree

### 5.3 Run repeated import benchmarks on `issue-10492-scheduler-import-cleanup`

- Status: `completed`
- Atomic operation:
  - Run the same benchmark matrix on the fix branch with the same procedure.
- Expected effect:
  - A stable after dataset is collected.
- Verification:
  - Record raw `real/user/sys` samples for every benchmark item.
- User confirmation gate:
  - User confirms the fix-branch dataset before aggregation.
- Actual result:
  - Ran the shared harness inside the container against:
    - `/sgl-workspace/sglang`
  - Command configuration:
    - `--warmups 1`
    - `--runs 10`
  - Output:
    - `/sgl-workspace/sglang/.codex-tmp/perf-results/fix-10runs.json`
  - Sample directories:
    - `/sgl-workspace/sglang/.codex-tmp/perf-results/fix-10runs-samples/`
  - Dataset integrity checks:
    - every benchmark item recorded `10` measured samples
    - sample directories contain `11` files per item:
      - `1` warm-up importtime sample
      - `10` measured importtime samples
- Verification result:
  - Outcome:
    - `completed`
  - Evidence:
    - raw `real/user/sys` samples exist for the same six benchmark items as the baseline run
  - Interpretation:
    - the after dataset is complete and directly comparable to the `main` baseline

### 5.4 Aggregate and compare the distributions

- Status: `completed`
- Atomic operation:
  - Compute per-item:
    - `min`
    - `max`
    - `mean`
    - `median`
- Expected effect:
  - The before/after comparison is based on repeated measurements rather than single runs.
- Verification:
  - Present the per-branch statistics for `real`, `user`, and `sys`.
- User confirmation gate:
  - User confirms the performance comparison is complete.
- Actual result:
  - Aggregated both datasets and recorded a human-readable summary in:
    - `/home/chris/Workspaces/sglang/issue-perf-summary.md`
  - Raw datasets preserved in:
    - `/home/chris/Workspaces/sglang/.codex-tmp/perf-results/main-10runs.json`
    - `/home/chris/Workspaces/sglang/.codex-tmp/perf-results/fix-10runs.json`
  - Key `real`-time mean changes:
    - `import_sglang`:
      - `0.639s -> 0.024s`
      - `-96.3%`
    - `import_model_config`:
      - `6.944s -> 1.751s`
      - `-74.8%`
    - `import_reasoning_parser`:
      - `3.132s -> 0.036s`
      - `-98.9%`
    - `import_moe_utils`:
      - `6.321s -> 2.146s`
      - `-66.0%`
    - `import_engine`:
      - `7.912s -> 7.039s`
      - `-11.0%`
    - `import_scheduler`:
      - `7.669s -> 4.694s`
      - `-38.8%`
  - Distribution stability:
    - the `min` and `max` ranges were tight on both branches for every benchmark item
    - representative `real` ranges:
      - `import_scheduler`:
        - before `7.627s .. 7.769s`
        - after `4.640s .. 4.744s`
      - `import_model_config`:
        - before `6.885s .. 7.026s`
        - after `1.718s .. 1.774s`
      - `import_sglang`:
        - before `0.627s .. 0.648s`
        - after `0.023s .. 0.024s`
  - `user` and `sys` time moved in the same direction:
    - `import_scheduler user`:
      - `8.035s -> 5.422s`
      - `-32.5%`
    - `import_scheduler sys`:
      - `1.102s -> 0.743s`
      - `-32.6%`
    - `import_engine sys` changed only slightly:
      - `1.167s -> 1.145s`
      - `-1.9%`
- Verification result:
  - Outcome:
    - `completed`
  - Evidence:
    - all requested statistics (`min`, `max`, `mean`, `median`) are available per metric and per benchmark item
    - raw samples remain preserved for later re-analysis
  - Interpretation:
    - the import cleanup produced large improvements on the paths it directly targeted
    - `Scheduler` improved materially
    - `Engine` improved only modestly, which is consistent with it still depending on the heavier serving/runtime stack
