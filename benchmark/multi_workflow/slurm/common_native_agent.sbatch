#!/usr/bin/env bash
#SBATCH --job-name=impactkv-native-agent
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --nodelist=gpu16
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

export IMPACTKV_PROJECT="$HOME/CodeMAS_Project/worktrees/sglang-common-agent"
export IMPACTKV_MODEL="$HOME/models/Qwen2.5-Coder-7B-Instruct"
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

backend="${IMPACTKV_COMMON_BACKEND:?set IMPACTKV_COMMON_BACKEND=cacheblend or kvcomm}"
mode="${IMPACTKV_COMMON_MODE:?set IMPACTKV_COMMON_MODE=dense or reuse}"
scope="${IMPACTKV_COMMON_SCOPE:-canary}"
instance="${IMPACTKV_COMMON_INSTANCE:-}"
model_dtype="${IMPACTKV_COMMON_DTYPE:-bfloat16}"
port="${IMPACTKV_COMMON_PORT:-$(python3 - <<'PY'
import socket

with socket.socket() as stream:
    stream.bind(("127.0.0.1", 0))
    print(stream.getsockname()[1])
PY
)}"
campaign="$IMPACTKV_ARTIFACTS/impactkv_common_agent_baselines_fresh24_20260812"

case "$scope" in
  canary)
    dataset="$campaign/canary_dataset"
    snapshot="$campaign/CANARY4.json"
    registration="$campaign/BRIDGE_CANARY4_REGISTRATION.json"
    ;;
  formal)
    dataset="$campaign/formal_dataset"
    snapshot="$campaign/FROZEN_FRESH24.json"
    registration="$campaign/BRIDGE_FRESH24_REGISTRATION.json"
    ;;
  *)
    echo "unknown common-agent scope: $scope" >&2
    exit 64
    ;;
esac

instance_key="${instance:-all}"
run_root="$campaign/runs/${scope}/${backend}_${mode}/${instance_key}"
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 73
fi
runtime="$IMPACTKV_RUNTIME_ROOT/logs/common-agent-$SLURM_JOB_ID"
mkdir -p "$runtime"

case "$backend" in
  cacheblend)
    backend_repo="$HOME/CodeMAS_Project/worktrees/cacheblend-common-agent"
    backend_python="$HOME/.venvs/cacheblend-native/bin/python"
    backend_command=(
      "$backend_python" "$backend_repo/example/common_agent_server.py"
      --model "$IMPACTKV_MODEL"
      --mode "$mode"
      --host 127.0.0.1
      --port "$port"
      --dtype "$model_dtype"
      --gpu-memory-utilization 0.90
      --max-model-len 32768
      --recompute-ratio 0.18
      --ledger "$runtime/BACKEND_LEDGER.jsonl"
    )
    backend_pythonpath="$backend_repo/vllm_blend:$backend_repo"
    ;;
  kvcomm)
    backend_repo="$HOME/CodeMAS_Project/worktrees/kvcomm-common-agent"
    backend_python="$HOME/.venvs/kvcomm-native/bin/python"
    backend_command=(
      "$backend_python" "$backend_repo/experiments/common_agent_server.py"
      --model "$IMPACTKV_MODEL"
      --mode "$mode"
      --host 127.0.0.1
      --port "$port"
      --threshold 0.3
      --window-size 5
      --ledger "$runtime/BACKEND_LEDGER.jsonl"
    )
    backend_pythonpath="$backend_repo"
    ;;
  *)
    echo "unknown backend: $backend" >&2
    exit 64
    ;;
esac

test -x "$backend_python"
test -f "$IMPACTKV_MODEL/model.safetensors.index.json"
printf '%q ' "${backend_command[@]}" >"$runtime/BACKEND_COMMAND.txt"
printf '\n' >>"$runtime/BACKEND_COMMAND.txt"

PYTHONNOUSERSITE=1 \
PYTHONPATH="$backend_pythonpath" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
KVFLOW_REPRO_DTYPE="$model_dtype" \
"${backend_command[@]}" >"$runtime/backend.log" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  if curl --connect-timeout 2 --max-time 5 -fsS \
    "http://127.0.0.1:$port/health" >"$runtime/health.json"; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  tail -n 200 "$runtime/backend.log" >&2 || true
  exit 70
fi

runner=(
  "$IMPACTKV_MINI_PYTHON"
  benchmark/multi_workflow/run_common_native_agent_experiment.py
  --backend "$backend"
  --mode "$mode"
  --backend-url "http://127.0.0.1:$port"
  --dataset "$dataset"
  --snapshot "$snapshot"
  --registration "$registration"
  --output "$run_root"
  --model "$IMPACTKV_MODEL"
  --step-limit 32
  --container-backend enroot
  --official
)
if [[ -n "$instance" ]]; then
  runner+=(--instance "$instance")
fi
if [[ -n "$instance" ]]; then
  runner_timeout="45m"
elif [[ "$scope" == "canary" ]]; then
  runner_timeout="3h"
else
  runner_timeout="11h"
fi
PYTHONPATH="$IMPACTKV_PROJECT" timeout --foreground --signal=TERM \
  --kill-after=60s "$runner_timeout" "${runner[@]}"

cp "$runtime/health.json" "$run_root/BACKEND_HEALTH.json"
cp "$runtime/BACKEND_COMMAND.txt" "$run_root/BACKEND_COMMAND.txt"
cp "$runtime/backend.log" "$run_root/BACKEND_SERVER.log"
if [[ -f "$runtime/BACKEND_LEDGER.jsonl" ]]; then
  cp "$runtime/BACKEND_LEDGER.jsonl" "$run_root/BACKEND_LEDGER.jsonl"
fi
