#!/usr/bin/env bash
#SBATCH --job-name=impactkv-exact-replay
#SBATCH --partition=long
#SBATCH --exclude=gpu[10-13,23-24]
#SBATCH --nodelist=gpu16
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

export IMPACTKV_PROJECT="$HOME/CodeMAS_Project/worktrees/sglang-common-agent"
export IMPACTKV_MODEL="$HOME/models/Qwen2.5-Coder-7B-Instruct"
source "$IMPACTKV_PROJECT/benchmark/multi_workflow/slurm/impactkv_home_env.sh"
cd "$IMPACTKV_PROJECT"

backend="${IMPACTKV_COMMON_BACKEND:?set IMPACTKV_COMMON_BACKEND}"
source_ledger="${IMPACTKV_COMMON_SOURCE_LEDGER:?set exact source BACKEND_LEDGER.jsonl}"
label="${IMPACTKV_COMMON_REPLAY_LABEL:-canary}"
limit="${IMPACTKV_COMMON_REPLAY_LIMIT:-}"
port="${IMPACTKV_COMMON_PORT:-$((35000 + SLURM_JOB_ID % 1000))}"
campaign="$IMPACTKV_ARTIFACTS/impactkv_common_agent_baselines_fresh24_20260812"
output="$campaign/exact_prompt_replay/${label}/${backend}"
runtime="$IMPACTKV_RUNTIME_ROOT/logs/common-exact-$SLURM_JOB_ID"
mkdir -p "$runtime"

case "$backend" in
  cacheblend)
    repo="$HOME/CodeMAS_Project/worktrees/cacheblend-common-agent"
    python="$HOME/.venvs/cacheblend-native/bin/python"
    command=(
      "$python" "$repo/example/common_agent_server.py"
      --model "$IMPACTKV_MODEL" --mode reuse --host 127.0.0.1 --port "$port"
      --dtype float16 --gpu-memory-utilization 0.85 --max-model-len 32768
      --recompute-ratio 0.18 --allow-benchmark-mode-override
      --ledger "$runtime/BACKEND_LEDGER.jsonl"
    )
    pythonpath="$repo/vllm_blend:$repo"
    ;;
  kvcomm)
    repo="$HOME/CodeMAS_Project/worktrees/kvcomm-common-agent"
    python="$HOME/.venvs/kvcomm-native/bin/python"
    command=(
      "$python" "$repo/experiments/common_agent_server.py"
      --model "$IMPACTKV_MODEL" --mode reuse --host 127.0.0.1 --port "$port"
      --threshold 0.3 --window-size 5 --allow-benchmark-mode-override
      --ledger "$runtime/BACKEND_LEDGER.jsonl"
    )
    pythonpath="$repo"
    ;;
  *)
    echo "unknown backend: $backend" >&2
    exit 64
    ;;
esac

test -f "$source_ledger"
test ! -e "$output"
PYTHONNOUSERSITE=1 \
PYTHONPATH="$pythonpath" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
KVFLOW_REPRO_DTYPE=float16 \
"${command[@]}" >"$runtime/backend.log" 2>&1 &
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
  if curl -fsS "http://127.0.0.1:$port/health" >"$runtime/health.json"; then
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
  benchmark/multi_workflow/run_common_exact_prompt_replay.py
  --backend "$backend"
  --backend-url "http://127.0.0.1:$port"
  --source-ledger "$source_ledger"
  --output "$output"
  --cycles 5
)
if [[ -n "$limit" ]]; then
  runner+=(--limit "$limit")
fi
PYTHONPATH="$IMPACTKV_PROJECT" "${runner[@]}"
cp "$runtime/health.json" "$output/BACKEND_HEALTH.json"
cp "$runtime/backend.log" "$output/BACKEND_SERVER.log"
