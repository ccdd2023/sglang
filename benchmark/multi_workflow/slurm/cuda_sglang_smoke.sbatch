#!/usr/bin/env bash
#SBATCH --job-name=impactkv-cuda-smoke
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
source "$HOME/CodeMAS_Project/sglang/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"
run_dir="$IMPACTKV_RUNTIME_ROOT/logs/cuda-smoke-$SLURM_JOB_ID"
mkdir -p "$run_dir"

"$IMPACTKV_EVAL_PYTHON" - <<'PY'
import json
from pathlib import Path
import os
import torch

x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
result = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "device": torch.cuda.get_device_name(0),
    "finite": bool(torch.isfinite(y).all().item()),
}
path = Path(os.environ["IMPACTKV_RUNTIME_ROOT"]) / "logs" / f"cuda-{os.environ['SLURM_JOB_ID']}.json"
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY

port="${IMPACTKV_SMOKE_PORT:-31000}"
"$IMPACTKV_EVAL_PYTHON" -m sglang.launch_server \
  --model-path "$IMPACTKV_MODEL" \
  --host 127.0.0.1 \
  --port "$port" \
  --context-length 32768 \
  --attention-backend triton \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --page-size 1 \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --enable-deterministic-inference \
  --watchdog-timeout 1200 \
  --random-seed 709609581 \
  >"$run_dir/server.log" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  if curl -fsS "http://127.0.0.1:$port/model_info" >"$run_dir/model_info.json"; then
    ready=1
    break
  fi
  sleep 2
done
test "$ready" = 1

curl -fsS --max-time 1800 "http://127.0.0.1:$port/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a Python function that adds two integers.","sampling_params":{"temperature":0,"max_new_tokens":32}}' \
  >"$run_dir/generate.json"
