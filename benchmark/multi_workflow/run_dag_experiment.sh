#!/bin/bash
# =============================================================================
# DAG Workflow KVFlow Benchmark - 实验运行脚本
#
# 运行 DAG workflow 实验，测试 Priority vs LRU 在 DAG 结构下的性能差异
#
# 使用方法:
#   ./run_dag_experiment.sh exp1        # 运行 exp1: 4 workflows, 1 轮
#   ./run_dag_experiment.sh exp2        # 运行 exp2: 4 workflows, 5 轮
#   ./run_dag_experiment.sh compare    # 运行hicache90k vs kvflow 对比
#   ./run_dag_experiment.sh ablation   # 运行消融实验
# =============================================================================

set -euo pipefail

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="/home/comp/25480812/CodeMAS_Project/sglang-kvflow"
BENCH_ROOT="$SGLANG_ROOT/benchmark/multi_workflow"
DAG_CONFIG="$BENCH_ROOT/dag_configs/diamond_6agent.json"
MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-dag-experiments/results"
mkdir -p "$LOG_DIR"

# 端口配置
LRU_PORT=30310
KVFLOW_PORT=30311

# 默认参数
NUM_WORKFLOWS=4
NUM_ROUNDS=3
WARMUP_ROUNDS=1
TIER0_LEN=512
TIER1_LEN=1024
TIER2_LEN=512
SUFFIX_LEN=64
OUTPUT_LEN=64

# Python 环境
export PATH="/home/comp/25480812/.conda/envs/sglang-kvflow/bin:$PATH"
export PYTHONPATH="$SGLANG_ROOT/python"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 等待服务器就绪
wait_for_server() {
    local port=$1
    local max_wait=120
    local waited=0
    
    log_info "等待服务器启动 (端口 $port)..."
    while [ $waited -lt $max_wait ]; do
        if curl -sf --noproxy '*' --max-time 1 "http://127.0.0.1:$port/health_generate" > /dev/null 2>&1; then
            log_success "服务器就绪 (等待 ${waited}s)"
            return 0
        fi
        if [ $((waited % 20)) -eq 0 ] && [ $waited -gt 0 ]; then
            log_info "仍在等待... (${waited}s)"
        fi
        sleep 1
        ((waited++))
    done
    
    log_error "服务器启动超时"
    return 1
}

# 停止服务器
stop_server() {
    local port=$1
    log_info "停止端口 $port 的服务器..."
    pid=$(lsof -ti :"$port" 2>/dev/null) || true
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        sleep 2
        log_info "服务器已停止"
    fi
}

# 启动 LRU 服务器 (hicache90k)
start_lru_server() {
    log_info "启动 LRU 服务器 (端口 $LRU_PORT)..."
    
    stop_server $LRU_PORT
    
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port $LRU_PORT \
        --tokenizer-path "$MODEL_PATH" \
        --tokenizer-mode auto \
        --trust-remote-code \
        \
        --mem-fraction-static 0.85 \
        --max-total-tokens 90000 \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        \
        --radix-eviction-policy lru \
        --enable-hierarchical-cache \
        --hicache-ratio 2.0 \
        --hicache-write-policy write_back \
        --hicache-io-backend direct \
        --hicache-mem-layout layer_first \
        \
        --attention-backend flashinfer \
        --sampling-backend flashinfer \
        \
        --tensor-parallel-size 2 \
        --disable-cuda-graph \
        \
        --log-level info \
        > "$LOG_DIR/server_lru_$LRU_PORT.log" 2>&1 &
    
    SERVER_PID=$!
    log_info "LRU 服务器 PID: $SERVER_PID"
}

# 启动 Priority 服务器 (kvflow)
start_kvflow_server() {
    log_info "启动 KVFlow 服务器 (端口 $KVFLOW_PORT)..."
    
    stop_server $KVFLOW_PORT
    
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port $KVFLOW_PORT \
        --tokenizer-path "$MODEL_PATH" \
        --tokenizer-mode auto \
        --trust-remote-code \
        \
        --mem-fraction-static 0.85 \
        --max-total-tokens 90000 \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 8192 \
        \
        --radix-eviction-policy priority \
        --enable-hierarchical-cache \
        --hicache-ratio 2.0 \
        --hicache-write-policy write_back \
        --enable-hicache-prefetch \
        --hicache-io-backend direct \
        --hicache-mem-layout layer_first \
        \
        --attention-backend flashinfer \
        --sampling-backend flashinfer \
        \
        --tensor-parallel-size 2 \
        --disable-cuda-graph \
        \
        --log-level info \
        > "$LOG_DIR/server_kvflow_$KVFLOW_PORT.log" 2>&1 &
    
    SERVER_PID=$!
    log_info "KVFlow 服务器 PID: $SERVER_PID"
}

# 运行 DAG 实验
run_dag_experiment() {
    local config=$1
    local port=$2
    local label=$3
    local extra_args="${4:-}"
    
    log_info "运行 $label 实验..."
    
    python "$BENCH_ROOT/bench_multi_workflow.py" \
        --config "$config" \
        --workflow-type dag \
        --dag-config "$DAG_CONFIG" \
        --num-workflows $NUM_WORKFLOWS \
        --tier0-len $TIER0_LEN \
        --tier1-len $TIER1_LEN \
        --tier2-len $TIER2_LEN \
        --suffix-len $SUFFIX_LEN \
        --output-len $OUTPUT_LEN \
        --num-rounds $NUM_ROUNDS \
        --warmup-rounds $WARMUP_ROUNDS \
        --host 127.0.0.1 \
        --port $port \
        --model "$MODEL_PATH" \
        --output-dir "$LOG_DIR" \
        --seed 42 \
        $extra_args
    
    log_success "$label 实验完成"
}

# 实验 1: 冒烟测试 (1 workflow, 1 round)
exp_smoke() {
    log_info "=============================================="
    log_info "实验: DAG 冒烟测试"
    log_info "=============================================="
    
    export NUM_WORKFLOWS=1
    export NUM_ROUNDS=2
    export WARMUP_ROUNDS=1
    
    # 只启动 kvflow 服务器
    start_kvflow_server
    wait_for_server $KVFLOW_PORT
    
    run_dag_experiment "kvflow" $KVFLOW_PORT "DAG 冒烟测试" "--baseline-json ''"
    
    stop_server $KVFLOW_PORT
}

# 实验 2: 基本对比 (4 workflows, 5 rounds)
exp_compare() {
    log_info "=============================================="
    log_info "实验: DAG - hicache90k vs kvflow 对比"
    log_info "=============================================="
    
    export NUM_WORKFLOWS=4
    export NUM_ROUNDS=5
    export WARMUP_ROUNDS=1
    
    # 启动两个服务器
    start_lru_server
    wait_for_server $LRU_PORT
    
    start_kvflow_server
    wait_for_server $KVFLOW_PORT
    
    # 运行 LRU 基线
    log_info ""
    log_info ">>> 运行 LRU 基线 (hicache90k) <<<"
    run_dag_experiment "hicache90k" $LRU_PORT "hicache90k"
    
    # 停止 LRU 服务器
    stop_server $LRU_PORT
    
    sleep 5
    
    # 运行 KVFlow
    log_info ""
    log_info ">>> 运行 KVFlow <<<"
    run_dag_experiment "kvflow" $KVFLOW_PORT "kvflow"
    
    # 停止 KVFlow 服务器
    stop_server $KVFLOW_PORT
    
    # 打印结果对比
    log_info ""
    log_info "=============================================="
    log_info "DAG 实验结果对比"
    log_info "=============================================="
    
    LRU_RESULT="$LOG_DIR/"$(ls -t "$LOG_DIR" | grep hicache90k | head -1)""
    KVFLOW_RESULT="$LOG_DIR/"$(ls -t "$LOG_DIR" | grep -E "^mwf_kvflow.*dag" | head -1)""
    
    if [ -f "$LRU_RESULT" ] && [ -f "$KVFLOW_RESULT" ]; then
        python3 -c "
import json
import sys

try:
    with open('$LRU_RESULT') as f:
        lru = json.load(f)
    with open('$KVFLOW_RESULT') as f:
        kv = json.load(f)
    
    lru_ttft = lru['aggregate']['stable_ttft_avg_ms']
    kv_ttft = kv['aggregate']['stable_ttft_avg_ms']
    speedup = lru_ttft / kv_ttft if kv_ttft > 0 else 0
    improvement = (lru_ttft - kv_ttft) / lru_ttft * 100 if lru_ttft > 0 else 0
    
    print(f'  hicache90k TTFT: {lru_ttft:.2f} ms')
    print(f'  kvflow TTFT:     {kv_ttft:.2f} ms')
    print(f'  Speedup:         {speedup:.2f}x')
    print(f'  Improvement:      {improvement:.1f}%')
except Exception as e:
    print(f'  Error comparing results: {e}')
"
    else
        log_warn "找不到结果文件进行对比"
        [ -f "$LRU_RESULT" ] && log_info "LRU 结果: $LRU_RESULT"
        [ -f "$KVFLOW_RESULT" ] && log_info "KVFlow 结果: $KVFLOW_RESULT"
    fi
}

# 实验 3: 压力测试 (8 workflows, 5 rounds)
exp_pressure() {
    log_info "=============================================="
    log_info "实验: DAG 高压力测试"
    log_info "=============================================="
    
    export NUM_WORKFLOWS=8
    export NUM_ROUNDS=5
    export WARMUP_ROUNDS=1
    
    # 启动两个服务器
    start_lru_server
    wait_for_server $LRU_PORT
    
    start_kvflow_server
    wait_for_server $KVFLOW_PORT
    
    # 运行 LRU 基线
    log_info ""
    log_info ">>> 运行 LRU 基线 (高压力) <<<"
    run_dag_experiment "hicache90k" $LRU_PORT "hicache90k-hp"
    
    # 停止 LRU 服务器
    stop_server $LRU_PORT
    
    sleep 5
    
    # 运行 KVFlow
    log_info ""
    log_info ">>> 运行 KVFlow (高压力) <<<"
    run_dag_experiment "kvflow" $KVFLOW_PORT "kvflow-hp"
    
    # 停止 KVFlow 服务器
    stop_server $KVFLOW_PORT
}

# 实验 4: 消融实验
exp_ablation() {
    log_info "=============================================="
    log_info "实验: DAG 消融实验 (Priority vs Prefetch)"
    log_info "=============================================="
    
    export NUM_WORKFLOWS=4
    export NUM_ROUNDS=5
    export WARMUP_ROUNDS=1
    
    # 测试配置列表
    configs=(
        "hicache90k:LRU基线"
        "priority_wb_only:Priority无Prefetch"
        "lru_wb_pf:LRU+Prefetch"
        "kvflow:Priority+Prefetch"
    )
    
    ports=(30320 30321 30322 30323)
    
    # 启动服务器并运行实验
    for i in "${!configs[@]}"; do
        IFS=':' read -r config label <<< "${configs[$i]}"
        port=${ports[$i]}
        
        log_info ""
        log_info ">>> ${label} (端口 $port) <<<"
        
        # 启动服务器
        if [ "$config" == "hicache90k" ] || [ "$config" == "lru_wb_pf" ]; then
            start_lru_server
        else
            start_kvflow_server
        fi
        
        # 重新配置端口
        stop_server $port
        sleep 2
        
        # 使用指定配置启动服务器
        if [ "$config" == "hicache90k" ]; then
            python -m sglang.launch_server \
                --model-path "$MODEL_PATH" \
                --port $port \
                --tokenizer-path "$MODEL_PATH" \
                --trust-remote-code \
                --mem-fraction-static 0.85 \
                --max-total-tokens 90000 \
                --radix-eviction-policy lru \
                --enable-hierarchical-cache \
                --hicache-ratio 2.0 \
                --hicache-write-policy write_back \
                --hicache-io-backend direct \
                --disable-cuda-graph \
                --log-level info \
                > "$LOG_DIR/server_${config}_${port}.log" 2>&1 &
        elif [ "$config" == "priority_wb_only" ]; then
            python -m sglang.launch_server \
                --model-path "$MODEL_PATH" \
                --port $port \
                --tokenizer-path "$MODEL_PATH" \
                --trust-remote-code \
                --mem-fraction-static 0.85 \
                --max-total-tokens 90000 \
                --radix-eviction-policy priority \
                --enable-hierarchical-cache \
                --hicache-ratio 2.0 \
                --hicache-write-policy write_back \
                --hicache-io-backend direct \
                --disable-cuda-graph \
                --log-level info \
                > "$LOG_DIR/server_${config}_${port}.log" 2>&1 &
        elif [ "$config" == "lru_wb_pf" ]; then
            python -m sglang.launch_server \
                --model-path "$MODEL_PATH" \
                --port $port \
                --tokenizer-path "$MODEL_PATH" \
                --trust-remote-code \
                --mem-fraction-static 0.85 \
                --max-total-tokens 90000 \
                --radix-eviction-policy lru \
                --enable-hierarchical-cache \
                --hicache-ratio 2.0 \
                --hicache-write-policy write_back \
                --enable-hicache-prefetch \
                --hicache-io-backend direct \
                --disable-cuda-graph \
                --log-level info \
                > "$LOG_DIR/server_${config}_${port}.log" 2>&1 &
        else  # kvflow
            python -m sglang.launch_server \
                --model-path "$MODEL_PATH" \
                --port $port \
                --tokenizer-path "$MODEL_PATH" \
                --trust-remote-code \
                --mem-fraction-static 0.85 \
                --max-total-tokens 90000 \
                --radix-eviction-policy priority \
                --enable-hierarchical-cache \
                --hicache-ratio 2.0 \
                --hicache-write-policy write_back \
                --enable-hicache-prefetch \
                --hicache-io-backend direct \
                --disable-cuda-graph \
                --log-level info \
                > "$LOG_DIR/server_${config}_${port}.log" 2>&1 &
        fi
        
        wait_for_server $port
        run_dag_experiment "$config" $port "$label"
        stop_server $port
        sleep 5
    done
    
    log_info ""
    log_info "=============================================="
    log_info "消融实验结果汇总"
    log_info "=============================================="
    
    python3 -c "
import json
import glob
import os

results = {}
for config, label in [
    ('hicache90k', 'LRU基线'),
    ('priority_wb_only', 'Priority无Prefetch'),
    ('lru_wb_pf', 'LRU+Prefetch'),
    ('kvflow', 'Priority+Prefetch'),
]:
    pattern = f'$LOG_DIR/mwf_{config}_*_dag*.json'
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if files:
        with open(files[0]) as f:
            data = json.load(f)
            ttft = data['aggregate']['stable_ttft_avg_ms']
            results[label] = ttft
            print(f'{label}: {ttft:.2f} ms')

if len(results) >= 2:
    baseline = results.get('LRU基线', 0)
    kvflow = results.get('Priority+Prefetch', 0)
    priority_only = results.get('Priority无Prefetch', 0)
    pf_only = results.get('LRU+Prefetch', 0)
    
    print()
    print('Speedup Analysis:')
    if baseline and kvflow:
        print(f'  kvflow vs LRU: {baseline/kvflow:.2f}x ({(baseline-kvflow)/baseline*100:.1f}%)')
    if baseline and priority_only:
        print(f'  Priority单独贡献: {baseline/priority_only:.2f}x ({(baseline-priority_only)/baseline*100:.1f}%)')
    if baseline and pf_only:
        print(f'  Prefetch单独贡献: {baseline/pf_only:.2f}x ({(baseline-pf_only)/baseline*100:.1f}%)')
"
}

# 打印帮助
show_help() {
    echo "DAG Workflow KVFlow Benchmark"
    echo ""
    echo "使用方法: $0 <实验类型>"
    echo ""
    echo "实验类型:"
    echo "  smoke      - 冒烟测试 (1 workflow, 2 rounds)"
    echo "  compare    - 基本对比 (hicache90k vs kvflow)"
    echo "  pressure   - 高压力测试 (8 workflows, 5 rounds)"
    echo "  ablation   - 消融实验 (4种配置对比)"
    echo "  all        - 运行所有实验"
    echo "  help       - 显示此帮助"
    echo ""
    echo "环境变量:"
    echo "  NUM_WORKFLOWS  - Workflow 数量 (默认: 4)"
    echo "  NUM_ROUNDS     - 轮数 (默认: 5)"
    echo ""
}

# 主函数
main() {
    local exp_type="${1:-help}"
    
    case "$exp_type" in
        smoke)
            exp_smoke
            ;;
        compare)
            exp_compare
            ;;
        pressure)
            exp_pressure
            ;;
        ablation)
            exp_ablation
            ;;
        all)
            exp_smoke
            exp_compare
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知实验类型: $exp_type"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
