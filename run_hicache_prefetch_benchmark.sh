#!/usr/bin/env bash
# =============================================================================
# HiCache + Prefetch Benchmark: 2-Card RTX 4090 (TP=2)
#
# Submits 4 SLURM jobs to specific RTX 4090 nodes (one job per node).
# Each job: start server -> wait ready -> benchmark.
#
# Configurations:
#   A: L2=90K + Priority + Prefetch   (no L2 pressure)
#   B: L2=90K + Priority + NO Prefetch (isolate prefetch effect)
#   C: L2=60K + Priority + Prefetch   (L2 pressure)
#   D: L2=60K + LRU + Prefetch        (LRU baseline)
#
# Working set: 20 agents × 4096 tokens = 81,920 tokens
# =============================================================================

set -eo pipefail

SGLKV_DIR="/home/comp/csgfyu/multi-agents/CodeGenerationKvCache/sglang-kvflow"
LOG_DIR="/home/comp/csgfyu/logs/sglang-kvflow"
PYTHON_BIN="${HOME}/miniconda3/envs/sglang-kvflow/bin/python"
CONDA_GXX="${HOME}/miniconda3/envs/sglang-kvflow/bin/g++"
MODEL_PATH="/home/comp/csgfyu/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
PARTITION="long"
GPUS_ALLOC=2
TIME_LIMIT="02:00:00"
NUM_AGENTS=20
PREFIX_LEN=4096
SUFFIX_LEN=64
OUTPUT_LEN=64
NUM_ROUNDS=5
WARMUP_ROUNDS=1
NUM_CONCURRENT=1
SERVER_PORT=30300

# Round-robin allocation: one job per clean RTX 4090 node
declare -A NODE_MAP=(
    ["A-pri-pref-l2-90k"]="gpu10"
    ["B-pri-nopref-l2-90k"]="gpu13"
    ["C-pri-pref-l2-60k"]="gpu14"
    ["D-lru-pref-l2-60k"]="gpu15"
)

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  HiCache + Prefetch Benchmark (2-GPU TP=2, Qwen3-8B, RTX 4090)"
echo "  Working set: ${NUM_AGENTS} × ${PREFIX_LEN} = $((NUM_AGENTS * PREFIX_LEN)) tokens"
echo "  L1 GPU capacity: ~60K tokens"
echo ""
echo "  A: L2=90K + Priority + Prefetch    (no pressure) -> gpu10"
echo "  B: L2=90K + Priority + NO Prefetch  (isolate prefetch) -> gpu13"
echo "  C: L2=60K + Priority + Prefetch     (L2 pressure) -> gpu14"
echo "  D: L2=60K + LRU + Prefetch          (LRU baseline) -> gpu15"
echo "============================================================"

submit_experiment() {
    local name="$1"
    local eviction="$2"
    local hicache_ratio="$3"
    local enable_prefetch="$4"
    local l2_size="$5"
    local target_node="${NODE_MAP[$name]}"

    if [ "$eviction" = "priority" ]; then
        local bench_config="priority"
    else
        local bench_config="baseline"
    fi

    local server_log="${LOG_DIR}/server-${name}.log"
    local bench_log="${LOG_DIR}/bench-${name}.log"

    echo ""
    echo "============================================================"
    echo "  Submitting: $name -> node=$target_node"
    echo "  eviction=$eviction, hicache_ratio=$hicache_ratio,"
    echo "  prefetch=$enable_prefetch, l2_size=$l2_size"
    echo "  bench_config=$bench_config"
    echo "============================================================"

    local hicache_flags=""
    if [ "$hicache_ratio" != "0" ]; then
        hicache_flags="--enable-hierarchical-cache --hicache-ratio ${hicache_ratio} --hicache-io-backend direct"
    fi

    local prefetch_flags=""
    if [ "$enable_prefetch" = "1" ]; then
        prefetch_flags="--enable-hicache-prefetch --enable-hicache-prefetch-log"
    fi

    local job_script="${LOG_DIR}/job-${name}.sh"

    # NOTE: 'OUTER_EOF' must be UNQUOTED so $name etc expand during heredoc creation
    cat > "${job_script}" << OUTER_EOF
#!/bin/bash
#SBATCH --job-name=sglang-${name}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${target_node}
#SBATCH --gres=gpu:${GPUS_ALLOC}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/sbatch-${name}.out
#SBATCH --error=${LOG_DIR}/sbatch-${name}.err

set -eo pipefail

SGLKV_DIR="${SGLKV_DIR}"
LOG_DIR="${LOG_DIR}"
PYTHON_BIN="${PYTHON_BIN}"
CONDA_GXX="${CONDA_GXX}"
MODEL_PATH="${MODEL_PATH}"
SERVER_PORT="${SERVER_PORT}"
GPUS_ALLOC="${GPUS_ALLOC}"
L2_SIZE="${l2_size}"
EVICTION="${eviction}"
HICACHE_FLAGS="${hicache_flags}"
PREFETCH_FLAGS="${prefetch_flags}"
BENCH_CONFIG="${bench_config}"
NUM_AGENTS="${NUM_AGENTS}"
PREFIX_LEN="${PREFIX_LEN}"
SUFFIX_LEN="${SUFFIX_LEN}"
OUTPUT_LEN="${OUTPUT_LEN}"
NUM_ROUNDS="${NUM_ROUNDS}"
WARMUP_ROUNDS="${WARMUP_ROUNDS}"
NUM_CONCURRENT="${NUM_CONCURRENT}"
NAME="${name}"
bench_log="${LOG_DIR}/bench-\${NAME}.log"
server_log="${LOG_DIR}/server-\${NAME}.log"

echo "=== Job Started: \${NAME} ==="
echo "Node: \$SLURM_JOB_NODELIST, GPUs: \$SLURM_JOB_GPUS"

# Start server
echo "[init] Starting SGLang server..."
cd "\${SGLKV_DIR}"
export PYTHONPATH="\${SGLKV_DIR}/python:\${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.5
export PATH="${CONDA_GXX%bin/g++}bin:${CUDA_HOME}/bin:\$PATH"
export CXX="\${CONDA_GXX}"
export CUDAHOSTCXX="\${CONDA_GXX}"
\${PYTHON_BIN} -m sglang.launch_server \
    --model-path "\${MODEL_PATH}" \
    --port \${SERVER_PORT} \
    --host 0.0.0.0 \
    --tp-size \${GPUS_ALLOC} \
    --mem-fraction-static 0.60 \
    --max-total-tokens \${L2_SIZE} \
    --radix-eviction-policy \${EVICTION} \
    --enable-cache-report \
    --disable-cuda-graph \
    --attention-backend triton \
    \${HICACHE_FLAGS} \
    \${PREFETCH_FLAGS} \
    > "\${server_log}" 2>&1 &
SERVER_PID=\$!
echo "[init] Server PID: \$SERVER_PID"

# Wait for server
echo "[init] Waiting for server..."
for i in \$(seq 1 120); do
    if curl -sf "http://127.0.0.1:\${SERVER_PORT}/health_generate" > /dev/null 2>&1; then
        echo "[init] Server ready (~=\$((i*5))s)"
        break
    fi
    sleep 5
    echo "[init] Waiting... (\$((i*5))s/600s)"
done

if ! curl -sf "http://127.0.0.1:\${SERVER_PORT}/health_generate" > /dev/null 2>&1; then
    echo "[init] ERROR: Server did not start"
    tail -30 "\${server_log}" 2>/dev/null || true
    kill \$SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run benchmark
echo "[init] Running benchmark with --config \${BENCH_CONFIG}..."
\${PYTHON_BIN} -m benchmark.priority.bench_priority \
    --config "\${BENCH_CONFIG}" \
    --host 127.0.0.1 \
    --port \${SERVER_PORT} \
    --model "\${MODEL_PATH}" \
    --num-agents \${NUM_AGENTS} \
    --prefix-len \${PREFIX_LEN} \
    --suffix-len \${SUFFIX_LEN} \
    --output-len \${OUTPUT_LEN} \
    --num-rounds \${NUM_ROUNDS} \
    --warmup-rounds \${WARMUP_ROUNDS} \
    --num-concurrent \${NUM_CONCURRENT} \
    --output-dir "\${LOG_DIR}" \
    --seed 42 \
    2>&1 | tee "\${bench_log}"

BENCH_EXIT=\${PIPESTATUS[0]}

echo "[cleanup] Stopping server..."
kill \$SERVER_PID 2>/dev/null || true
wait \$SERVER_PID 2>/dev/null || true

echo "[done] Exit code: \$BENCH_EXIT, Log: \${bench_log}"
echo "=== Job Finished ==="
exit \$BENCH_EXIT
OUTER_EOF

    chmod +x "${job_script}"
    sbatch "${job_script}"
    echo "[submit] Submitted: $name -> $target_node"
}

# A: L2=90K + Priority + Prefetch
submit_experiment "A-pri-pref-l2-90k" "priority" "1.5" "1" "90000"

# B: L2=90K + Priority + NO Prefetch
submit_experiment "B-pri-nopref-l2-90k" "priority" "1.5" "0" "90000"

# C: L2=60K + Priority + Prefetch
submit_experiment "C-pri-pref-l2-60k" "priority" "1.0" "1" "60000"

# D: L2=60K + LRU + Prefetch
submit_experiment "D-lru-pref-l2-60k" "lru" "1.0" "1" "60000"

echo ""
echo "============================================================"
echo "  All 4 experiments submitted!"
echo "  Check: squeue -u $USER | grep sglang"
echo "  Logs: ${LOG_DIR}/bench-{A,B,C,D}-*.log"
echo "============================================================"
echo ""
echo "Key comparisons:"
echo "  A vs B: proactive prefetch effect (L2=90K, no pressure)"
echo "  C vs D: eviction policy effect (L2=60K, with L2 pressure)"
echo "  A vs C: L2 capacity pressure on Priority"
