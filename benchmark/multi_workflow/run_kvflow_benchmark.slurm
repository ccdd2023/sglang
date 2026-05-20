#!/bin/bash
# =============================================================================
# KVFlow Benchmark - Slurm Job Script
#
# This script runs the KVFlow multi-workflow benchmark on a GPU cluster
# using Slurm job scheduling.
#
# Usage:
#   sbatch run_kvflow_benchmark.slurm              # Run all experiments
#   sbatch --export=EXP=quick run_kvflow_benchmark.slurm  # Quick smoke test
#
# Environment variables:
#   EXP         - Experiment to run: quick, exp1, exp2, exp3, all (default: all)
#   MODEL_PATH  - Path to the model (default: auto-download Qwen3-1.7B)
#   PARTITION   - Slurm partition (default: short)
#   TIME_LIMIT  - Job time limit (default: 04:00:00)
#
# =============================================================================

#SBATCH --job-name=kvflow-bench
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kvflow_%A_%a_%j.out
#SBATCH --error=logs/kvflow_%A_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@%h

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Experiment selection
EXP="${EXP:-all}"  # quick, exp1, exp2, exp3, all

# Paths - auto-detect based on current location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
SGLANG_ROOT_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow"

# Log directories
LOG_DIR="${LOG_DIR:-/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow}"
RESULT_DIR="$LOG_DIR/results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Conda configuration
CONDA_ROOT="/usr/local/miniconda/py312_24.7.1-0"
CONDA_ENV_NAME="sglang-kvflow"
CONDA_ENV_PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

# Model configuration - use smaller model for testing
if [[ -z "${MODEL_PATH:-}" ]]; then
    # Use smaller model for quick testing
    MODEL_PATH="${MODEL_PATH:-/home/comp/25480812/models/Qwen3-1.7B}"
    HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3-1.7B}"
fi

# Model download location
MODEL_DIR="/home/comp/25480812/models"
mkdir -p "$MODEL_DIR"

# Server ports
PORT_BASE="${PORT_BASE:-30000}"
PORT_LRU=$((PORT_BASE))
PORT_KVFLOW=$((PORT_BASE + 1))

# GPU memory wait threshold (MiB)
GPU_MEM_THRESHOLD="${GPU_MEM_THRESHOLD:-4000}"

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$EXP] $*"
}

log_section() {
    log "=========================================="
    log "$1"
    log "=========================================="
}

kill_port() {
    local p="$1"
    if pid=$(lsof -ti :"$p" 2>/dev/null); then
        log "Killing port $p (PID=$pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 8
    fi
}

wait_ready() {
    local port="$1"
    local max_wait="${2:-90}"
    log "Waiting for server on port $port..."
    for i in $(seq 1 "$max_wait"); do
        if curl -sf --max-time 2 "http://127.0.0.1:$port/health_generate" > /dev/null 2>&1; then
            log "Server on port $port ready after $((i*5))s!"
            return 0
        fi
        [ $((i % 6)) -eq 0 ] && log "  ...still waiting, $((i*5))s..."
        sleep 5
    done
    log "ERROR: Server on port $port did not start"
    tail -20 "$LOG_DIR"/server_*.log 2>/dev/null || true
    return 1
}

wait_gpu_free() {
    local max_wait="${1:-600}"
    log "Waiting for GPU memory to free (threshold: ${GPU_MEM_THRESHOLD}MiB)..."
    for i in $(seq 1 "$max_wait"); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [[ -n "$used" ]] && [[ "$used" -lt "$GPU_MEM_THRESHOLD" ]]; then
            log "GPU memory freed: ${used}MiB"
            return 0
        fi
        [ $((i % 12)) -eq 0 ] && log "  GPU memory still: ${used:-N/A}MiB..."
        sleep 5
    done
    log "WARNING: GPU memory still high after ${max_wait}s wait"
    return 1
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_conda_env() {
    log_section "Setting up Conda Environment"
    
    # Check if conda is available
    if [[ ! -f "$CONDA_SH" ]]; then
        log "ERROR: Conda not found at $CONDA_SH"
        return 1
    fi
    
    source "$CONDA_SH"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log "Environment '$CONDA_ENV_NAME' already exists"
        
        # Check if sglang is installed
        if python -c "import sglang" 2>/dev/null; then
            log "SGLang is already installed"
        else
            log "SGLang not installed, installing now..."
            install_sglang
        fi
    else
        log "Creating conda environment '$CONDA_ENV_NAME'..."
        conda create -n "$CONDA_ENV_NAME" python=3.12 -y
        conda activate "$CONDA_ENV_NAME"
        install_sglang
    fi
    
    log "Environment setup complete"
}

install_sglang() {
    log "Installing SGLang from source..."
    
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
    
    # Install PyTorch with CUDA 12.1
    log "Installing PyTorch with CUDA 12.1..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall || true
    
    # Reinstall sgl-kernel with correct CUDA version
    log "Reinstalling sgl-kernel for CUDA 12.1..."
    pip uninstall sgl-kernel -y || true
    pip install sgl-kernel==0.2.5 --force-reinstall || true
    
    # Install SGLang from source
    log "Installing SGLang..."
    cd "$SGLANG_ROOT_DIR/python"
    pip install -e . --no-deps --force-reinstall || true
    
    # Install dependencies
    log "Installing dependencies..."
    pip install aiohttp transformers sentencepiece || true
    
    # Verify installation
    if python -c "import sglang; print('SGLang version:', sglang.__version__)"; then
        log "SGLang installed successfully"
    else
        log "WARNING: SGLang import failed, but continuing..."
    fi
    
    cd "$BENCHMARK_DIR"
}

download_model() {
    log_section "Checking Model"
    
    # Check if model exists (handle HuggingFace Hub format)
    local MODEL_FOUND=0
    local ACTUAL_MODEL_PATH=""
    
    # Try direct path
    if [[ -d "$MODEL_PATH" ]]; then
        MODEL_FOUND=1
        ACTUAL_MODEL_PATH="$MODEL_PATH"
        log "Model found at direct path: $MODEL_PATH"
    fi
    
    # Try HuggingFace Hub format paths
    if [[ $MODEL_FOUND -eq 0 ]]; then
        local HF_MODEL_NAME
        HF_MODEL_NAME=$(basename "$HF_MODEL_ID")
        local HF_PATHS=(
            "$MODEL_DIR/hub/models--${HF_MODEL_ID//\//--}"
            "$MODEL_DIR/hub/models--${HF_MODEL_NAME}--*"
        )
        for p in "${HF_PATHS[@]}"; do
            local MATCH
            MATCH=$(find "$MODEL_DIR" -maxdepth 3 -type d -name "models--${HF_MODEL_ID//\//--}" 2>/dev/null | head -1)
            if [[ -n "$MATCH" ]]; then
                MODEL_FOUND=1
                ACTUAL_MODEL_PATH="$MATCH/snapshots"/*  # Use glob to get actual snapshot
                break
            fi
        done
    fi
    
    if [[ $MODEL_FOUND -eq 1 ]]; then
        log "Model already exists at: $ACTUAL_MODEL_PATH"
        # Update MODEL_PATH to actual path
        MODEL_PATH="$ACTUAL_MODEL_PATH"
        return 0
    fi
    
    log "Downloading model: $HF_MODEL_ID"
    log "This may take a while for large models..."
    
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
    
    export HF_HOME="$MODEL_DIR"
    mkdir -p "$HF_HOME"
    
    # Download model and tokenizer
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('$HF_MODEL_ID', trust_remote_code=True)
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('$HF_MODEL_ID', trust_remote_code=True)
print('Download complete!')
"
    
    # Verify download and find actual path
    local DOWNLOADED_PATH
    DOWNLOADED_PATH=$(find "$MODEL_DIR" -maxdepth 4 -type d -name "models--${HF_MODEL_ID//\//--}" 2>/dev/null | head -1)
    if [[ -n "$DOWNLOADED_PATH" ]]; then
        log "Model downloaded successfully to: $DOWNLOADED_PATH"
        MODEL_PATH="$DOWNLOADED_PATH/snapshots"/*  # Update to actual path
    else
        log "ERROR: Model download failed"
        return 1
    fi
}

# =============================================================================
# Server Management
# =============================================================================

start_server() {
    local config="$1"
    local port="$2"
    local log_file="$LOG_DIR/server_${config}_${port}.log"

    log "Starting server: config=$config, port=$port"
    kill_port "$port"

    local RADIX="lru"
    local HICACHE_FLAGS=""
    local MAX_TOKENS="--max-total-tokens 60000"
    local MEM_FRAC="0.85"

    if [[ "$config" == "kvflow" ]]; then
        RADIX="priority"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.5 --hicache-write-policy write_back --enable-hicache-prefetch"
        MAX_TOKENS="--max-total-tokens 90000"
    elif [[ "$config" == "hicache" ]]; then
        RADIX="lru"
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through"
    fi

    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
    
    cd "$SGLANG_ROOT_DIR"

    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$port" \
        --tp-size 2 \
        --mem-fraction-static "$MEM_FRAC" \
        $MAX_TOKENS \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        --radix-eviction-policy "$RADIX" \
        $HICACHE_FLAGS \
        --hicache-io-backend direct \
        --enable-cache-report \
        --disable-cuda-graph \
        --log-level info \
        > "$log_file" 2>&1 &

    log "Server PID=$! (log: $log_file)"
    echo "$!" > "$LOG_DIR/.server_${config}_${port}.pid"
}

stop_server() {
    local config="$1"
    local port="$2"
    kill_port "$port"
    rm -f "$LOG_DIR/.server_${config}_${port}.pid"
    sleep 20
}

# =============================================================================
# Benchmark Runner
# =============================================================================

run_benchmark() {
    local config="$1"
    local port="$2"
    shift 2
    local extra_args="$@"
    local bench_log="$LOG_DIR/bench_${config}_${port}.log"

    log "Running $config benchmark on port $port..."
    log "Extra args: $extra_args"

    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
    cd "$SGLANG_ROOT_DIR"

    python -m benchmark.multi_workflow.bench_multi_workflow \
        --config "$config" \
        --host 127.0.0.1 \
        --port "$port" \
        --agents-seed 42 \
        --seed 42 \
        --output-dir "$RESULT_DIR" \
        $extra_args \
        2>&1 | tee "$bench_log"
}

# =============================================================================
# Experiment Definitions
# =============================================================================

run_exp_quick() {
    log_section "QUICK SMOKE TEST"

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 90 || return 1
    sleep 5
    run_benchmark "hicache" "$PORT_LRU" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 3 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 5

    wait_gpu_free 120
    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 90 || return 1
    sleep 5
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 3 --warmup-rounds 1 \
        --baseline-json "$(ls -t "$RESULT_DIR"/mwf_hicache_*.json 2>/dev/null | head -1)"
    stop_server "kvflow" "$PORT_KVFLOW"

    log "QUICK TEST COMPLETE"
}

run_exp1_single() {
    log_section "EXPERIMENT 1: Single Workflow (Light Pressure)"

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 90 || return 1
    sleep 5
    run_benchmark "hicache" "$PORT_LRU" \
        --num-workflows 1 --agents-per-workflow 4 \
        --shared-p-len 2048 --unique-p-len 512 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 5

    wait_gpu_free 120
    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 90 || return 1
    sleep 5
    local LRU_JSON
    LRU_JSON=$(ls -t "$RESULT_DIR"/mwf_hicache_*1wf*.json 2>/dev/null | head -1 || true)
    local BASELINE_ARG=""
    [[ -n "$LRU_JSON" ]] && BASELINE_ARG="--baseline-json $LRU_JSON"
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 1 --agents-per-workflow 4 \
        --shared-p-len 2048 --unique-p-len 512 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_ARG
    stop_server "kvflow" "$PORT_KVFLOW"

    log "EXP 1 COMPLETE"
}

run_exp2_multi() {
    log_section "EXPERIMENT 2: 4 Concurrent Workflows (Moderate Pressure)"

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 90 || return 1
    sleep 5
    run_benchmark "hicache" "$PORT_LRU" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 512 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1
    stop_server "hicache" "$PORT_LRU"
    sleep 5

    wait_gpu_free 120
    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 90 || return 1
    sleep 5
    local LRU_JSON
    LRU_JSON=$(ls -t "$RESULT_DIR"/mwf_hicache_*4wf*.json 2>/dev/null | head -1 || true)
    local BASELINE_ARG=""
    [[ -n "$LRU_JSON" ]] && BASELINE_ARG="--baseline-json $LRU_JSON"
    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 4 --agents-per-workflow 5 \
        --shared-p-len 2048 --unique-p-len 512 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        $BASELINE_ARG
    stop_server "kvflow" "$PORT_KVFLOW"

    log "EXP 2 COMPLETE"
}

run_exp3_heavy() {
    log_section "EXPERIMENT 3: 8 Concurrent Workflows (Heavy Pressure)"

    start_server "hicache" "$PORT_LRU"
    wait_ready "$PORT_LRU" 90 || return 1
    sleep 3

    start_server "kvflow" "$PORT_KVFLOW"
    wait_ready "$PORT_KVFLOW" 90 || return 1
    sleep 3

    run_benchmark "hicache" "$PORT_LRU" \
        --num-workflows 8 --agents-per-workflow 8 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1

    run_benchmark "kvflow" "$PORT_KVFLOW" \
        --num-workflows 8 --agents-per-workflow 8 \
        --shared-p-len 2048 --unique-p-len 1024 \
        --suffix-len 64 --output-len 64 \
        --num-rounds 5 --warmup-rounds 1 \
        --baseline-json "$(ls -t "$RESULT_DIR"/mwf_hicache_*64agents*8wf*.json 2>/dev/null | head -1)"

    stop_server "hicache" "$PORT_LRU"
    stop_server "kvflow" "$PORT_KVFLOW"

    log "EXP 3 COMPLETE"
}

# =============================================================================
# Main
# =============================================================================

print_summary() {
    log_section "RESULTS SUMMARY"
    log ""
    log "Results: $RESULT_DIR"
    log "Logs:    $LOG_DIR"
    log ""
    log "JSON files:"
    ls -lh "$RESULT_DIR"/mwf_*.json 2>/dev/null | awk '{print "  "$NF}' || echo "  (none yet)"
    log ""
    log "To analyze results, run:"
    log "  python -m benchmark.multi_workflow.analyze_results $RESULT_DIR"
}

usage() {
    cat <<'EOF'
Usage: sbatch [options] run_kvflow_benchmark.slurm

Options:
  --export=EXP=quick    Run specific experiment (quick, exp1, exp2, exp3, all)
  --partition=short     Use short partition

Environment:
  MODEL_PATH=/path/to/model    Path to model (auto-download if not set)
  HF_MODEL_ID=Qwen/Qwen3-1.7B HuggingFace model ID to download
EOF
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "KVFlow Benchmark Starting"
log "=========================================="
log "Experiment: $EXP"
log "Model: $MODEL_PATH"
log "Log dir: $LOG_DIR"
log "Result dir: $RESULT_DIR"
log "=========================================="

# Setup environment and install dependencies
setup_conda_env

# Download model if needed
download_model

# Run the experiment
case "$EXP" in
    exp1)   run_exp1_single ;;
    exp2)   run_exp2_multi ;;
    exp3)   run_exp3_heavy ;;
    quick)  run_exp_quick ;;
    all)    run_exp1_single && run_exp2_multi && run_exp3_heavy ;;
    help|--help|-h) usage; exit 0 ;;
    *)      log "Unknown experiment: $EXP"; usage; exit 1 ;;
esac

print_summary
log "JOB COMPLETE"
