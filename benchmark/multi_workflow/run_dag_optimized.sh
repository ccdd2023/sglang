#!/bin/bash
###############################################################################
# DAG Optimized KVFlow 消融验证实验
#
# 实验目的：验证 DAG-aware Priority + Prefetch 锁定优化后的效果
#
# 测试配置（4 种）：
#   lru_wb_only:        LRU 基线（无 Priority，无 Prefetch）
#   priority_dag:       Priority + DAG-aware 汇聚保护（无 Prefetch）
#   priority_pf_lock:   Priority + DAG-aware + Prefetch 锁定（完整优化）
#   kvflow:              完整 KVFlow (Priority v3 + Prefetch Lock)
#
# 测试压力：
#   low:  4 workflows × 5 agents = 20 agents
#   high: 16 workflows × 5 agents = 80 agents
#
# 预期结果：
#   - Priority DAG 应优于原始 Priority（+5.9%~+14.4% 基础上提升）
#   - KVFlow Optimized 应消除负面交互，接近 Priority × Prefetch 的乘积效果
###############################################################################

set -uo pipefail

# ============================================================================
# 配置
# ============================================================================
SGLANG_DIR="/home/comp/25480812/CodeMAS_Project/sglang-kvflow"
BENCH_DIR="${SGLANG_DIR}/benchmark/multi_workflow"
LOG_DIR="/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow"
PYTHON_BIN="/home/comp/25480812/.conda/envs/sglang-kvflow/bin/python"

# 确保 PYTHONPATH 包含 sglang-kvflow/python（供 benchmark 模块使用）
export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"

# 模型配置
MODEL_PATH="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B"
TP_SIZE=2

# 实验配置
# 格式：bench_config_name:server_evict:hicache_enabled:prefetch_enabled:port
OPT_CONFIGS=(
    "lru_wb_only:lru:true:false:30420"
    "lru_wb_pf:lru:true:true:30421"
    "priority_dag:priority:true:false:30422"
    "kvflow:priority:true:true:30423"
)

# ============================================================================
# 辅助函数
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

usage() {
    cat <<EOF
用法: $0 <low|high|all>

参数:
    low     运行低压力测试 (4 workflows)
    high    运行高压力测试 (16 workflows)
    all     运行全部测试 (low + high)

示例:
    $0 all      # 运行全部实验
    $0 low      # 仅运行低压力测试
EOF
}

# ============================================================================
# 服务器管理
# ============================================================================

wait_server_ready() {
    local url=$1
    local port=$(echo "$url" | sed 's/.*://')
    local timeout=${2:-120}
    local elapsed=0
    log "等待服务器启动: $url"

    # 使用 Python 检测 HTTP（避免 curl 代理问题）
    while [ $elapsed -lt $timeout ]; do
        # 使用 Python 检测端口和 HTTP 响应
        http_code=$("$PYTHON_BIN" -c "
import urllib.request
import urllib.error
import sys
try:
    req = urllib.request.Request('${url}/v1/models')
    req.set_proxy = None  # 禁用代理
    # 使用 opener without proxy
    import os
    env_backup = {}
    for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        env_backup[k] = os.environ.pop(k, None)
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        print(resp.getcode())
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v
except Exception as e:
    print(0)
" 2>/dev/null)

        if [[ "$http_code" == "200" ]]; then
            log "服务器就绪: $url"
            return 0
        fi

        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    log "服务器启动超时: $url"
    return 1
}

start_server() {
    # 禁用代理，避免 ProxyError
    export http_proxy=""
    export https_proxy=""
    export HTTP_PROXY=""
    export HTTPS_PROXY=""

    local bench_config=$1
    local evict=$2
    local hicache_enabled=$3
    local enable_prefetch=$4
    local port=$5
    local log_file="${LOG_DIR}/server_${bench_config}_${port}.log"

    log "启动服务器: ${bench_config} (evict=${evict}, hicache=${hicache_enabled}, prefetch=${enable_prefetch}, port=${port})"

    local SERVER_FLAGS=""
    SERVER_FLAGS="--port ${port}"
    SERVER_FLAGS="${SERVER_FLAGS} --model-path ${MODEL_PATH}"
    SERVER_FLAGS="${SERVER_FLAGS} --tokenizer-path ${MODEL_PATH}"
    SERVER_FLAGS="${SERVER_FLAGS} --trust-remote-code"
    SERVER_FLAGS="${SERVER_FLAGS} --mem-fraction-static 0.88"
    SERVER_FLAGS="${SERVER_FLAGS} --radix-eviction-policy ${evict}"
    SERVER_FLAGS="${SERVER_FLAGS} --enable-hierarchical-cache"
    SERVER_FLAGS="${SERVER_FLAGS} --hicache-ratio 1.5"
    SERVER_FLAGS="${SERVER_FLAGS} --hicache-write-policy write_back"
    SERVER_FLAGS="${SERVER_FLAGS} --hicache-io-backend direct"

    if [[ "$enable_prefetch" == "true" ]]; then
        SERVER_FLAGS="${SERVER_FLAGS} --enable-hicache-prefetch"
    fi

    mkdir -p "${LOG_DIR}"
    cd "${SGLANG_DIR}"

    nohup ${PYTHON_BIN} -m sglang.launch_server ${SERVER_FLAGS} > "${log_file}" 2>&1 &
    SERVER_PID=$!

    log "服务器 PID: ${SERVER_PID}, 日志: ${log_file}"

    # 等待服务器就绪
    wait_server_ready "http://127.0.0.1:${port}" 180 || {
        log "服务器启动失败"
        kill ${SERVER_PID} 2>/dev/null || true
        return 1
    }

    # PID 输出到 stderr（避免与 log() 输出混淆）
    echo "SERVER_PID:${SERVER_PID}" >&2
}

stop_server() {
    local pid=$1
    log "停止服务器 PID: ${pid}"
    kill ${pid} 2>/dev/null || true
    sleep 5
    # 强制终止（如果还在运行）
    kill -9 ${pid} 2>/dev/null || true
    sleep 3

    # 强制清理残留的 sglang 进程
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang/srt" 2>/dev/null || true
    sleep 5
    # 强制清理 GPU 内存
    "$PYTHON_BIN" -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()" 2>/dev/null || true
    sleep 10
}

# ============================================================================
# 运行基准测试
# ============================================================================

run_bench() {
    local bench_config=$1
    local port=$2
    shift 2
    local bench_log="${LOG_DIR}/bench_${bench_config}_${port}.log"

    log "运行基准测试: ${bench_config} @ port ${port} (args: $*)"

    # 切换到正确目录并设置 PYTHONPATH（start_server 会 cd 到 SGLANG_DIR）
    (
        cd "${SGLANG_DIR}"
        export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"
        ${PYTHON_BIN} -m benchmark.multi_workflow.bench_multi_workflow \
            --config "${bench_config}" \
            --host "127.0.0.1" \
            --port "${port}" \
            --output-len 32 \
            --suffix-len 32 \
            "$@" \
            2>&1 | tee "${bench_log}"
    )

    log "基准测试完成: ${bench_config} @ port ${port}"
}

# ============================================================================
# 运行单个压力级别
# ============================================================================

run_pressure_level() {
    local pressure=$1  # "low" 或 "high"

    if [[ "$pressure" == "low" ]]; then
        local num_wf=4
        local extra_desc="(4 workflows)"
    else
        local num_wf=16
        local extra_desc="(16 workflows)"
    fi

    log "========================================"
    log "开始 ${pressure} 压力测试 ${extra_desc}"
    log "========================================"

    # 串行运行：每次启动一个服务器，运行基准测试，然后停止
    # 避免同时占用过多内存（4 服务器 × ~145GB > 256GB 会 OOM）
    for config in "${OPT_CONFIGS[@]}"; do
        IFS=':' read -r bench_config evict hicache pf port <<< "${config}"

        # 启动服务器（允许失败，跳过）- 从 stderr 解析 PID
        local pid_output
        pid_output=$(start_server "${bench_config}" "${evict}" "${hicache}" "${pf}" "${port}" 2>&1) || {
            log "WARNING: 服务器启动失败: ${pid_output}"
            continue
        }
        pid=$(echo "$pid_output" | grep "SERVER_PID:" | sed 's/SERVER_PID://') || {
            log "WARNING: 无法解析 PID from: ${pid_output}"
            continue
        }

        # 运行基准测试（串行）
        run_bench "${bench_config}" "${port}" \
            --workflow-type dag \
            --dag-config "${BENCH_DIR}/configs/dag_parallel_dev.json" \
            --num-workflows ${num_wf} \
            --tier0-len 2048 \
            --tier1-len 1024 \
            --tier2-len 1024 \
            --num-rounds 5 \
            --warmup-rounds 1 \
            --agents-seed 42 || log "WARNING: 基准测试失败 ${bench_config}"

        # 停止服务器
        stop_server ${pid}

        # 等待 GPU 内存回收
        log "等待 GPU 内存回收..."
        sleep 15
    done

    log "========================================"
    log "${pressure} 压力测试完成"
    log "========================================"
}

# ============================================================================
# 分析结果
# ============================================================================

analyze_results() {
    log "分析实验结果..."

    ${PYTHON_BIN} << 'PYEOF'
import json
import glob
import os
from collections import defaultdict

LOG_DIR = "/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow"

# 查找最新的结果文件
patterns = [
    "mwf_lru_wb_only_*_4wf.json",
    "mwf_priority_dag_*_4wf.json",
    "mwf_priority_pf_lock_*_4wf.json",
    "mwf_kvflow_*_4wf.json",
    "mwf_lru_wb_only_*_16wf.json",
    "mwf_priority_dag_*_16wf.json",
    "mwf_priority_pf_lock_*_16wf.json",
    "mwf_kvflow_*_16wf.json",
]

configs = {
    "lru_wb_only": {},
    "priority_dag": {},
    "priority_pf_lock": {},
    "kvflow": {},
}

for pattern in patterns:
    files = glob.glob(os.path.join(LOG_DIR, "results", pattern))
    for f in files:
        fname = os.path.basename(f)
        for cfg_name in configs:
            if fname.startswith(f"mwf_{cfg_name}_"):
                if "16wf" in fname:
                    pressure = "high"
                else:
                    pressure = "low"
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    configs[cfg_name][pressure] = data
                except:
                    pass

print("\n" + "="*80)
print("DAG Optimized KVFlow Ablation Results")
print("="*80)

for pressure in ["low", "high"]:
    wf_count = "4 workflows" if pressure == "low" else "16 workflows"
    print(f"\n### {wf_count}")

    rows = []
    for cfg_name, cfg_data in configs.items():
        if pressure not in cfg_data:
            continue
        data = cfg_data[pressure]
        agg = data.get("aggregate", {})
        rs = data.get("round_summaries", {})

        stable_ttft = agg.get("avg_ttft", 0)
        hit_rate = agg.get("est_hit_rate", 0)
        warmup_ttft = agg.get("warmup_ttft", 0)

        rows.append({
            "config": cfg_name,
            "stable_ttft": stable_ttft,
            "hit_rate": hit_rate,
            "warmup_ttft": warmup_ttft,
        })

    # 计算相对 LRU 基线的 speedup
    lru_row = next((r for r in rows if r["config"] == "lru_wb_only"), None)
    if lru_row:
        lru_ttft = lru_row["stable_ttft"]
        print(f"\n{'Config':<20} {'TTFT (ms)':<12} {'vs LRU':<10} {'Hit Rate':<10} {'Warmup (ms)':<12}")
        print("-" * 70)
        for row in rows:
            if lru_ttft > 0:
                speedup = (lru_ttft - row["stable_ttft"]) / lru_ttft * 100
                speedup_str = f"+{speedup:.1f}%" if speedup >= 0 else f"{speedup:.1f}%"
            else:
                speedup_str = "N/A"
            print(f"{row['config']:<20} {row['stable_ttft']:<12.1f} {speedup_str:<10} "
                  f"{row['hit_rate']*100:<10.1f}% {row['warmup_ttft']:<12.1f}")

        # 组件贡献分析
        pri_row = next((r for r in rows if r["config"] == "priority_dag"), None)
        pf_row = next((r for r in rows if r["config"] == "priority_pf_lock"), None)
        full_row = next((r for r in rows if r["config"] == "kvflow"), None)

        if lru_row and pri_row and pf_row and full_row:
            print("\n  Component Analysis:")
            print(f"  - Priority DAG contribution:    {(lru_row['stable_ttft'] - pri_row['stable_ttft'])/lru_row['stable_ttft']*100:+.1f}%")
            print(f"  - Prefetch contribution:       {(pri_row['stable_ttft'] - pf_row['stable_ttft'])/pri_row['stable_ttft']*100:+.1f}%")
            if pf_row['stable_ttft'] > 0:
                interaction = (pf_row['stable_ttft'] - full_row['stable_ttft']) / pf_row['stable_ttft'] * 100
                print(f"  - Interaction effect:          {interaction:+.1f}%")
            print(f"  - Total KVFlow Optimized:      {(lru_row['stable_ttft'] - full_row['stable_ttft'])/lru_row['stable_ttft']*100:+.1f}%")

print("\n" + "="*80)
PYEOF

    log "分析完成"
}

# ============================================================================
# 主流程
# ============================================================================

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

case "$1" in
    low)
        run_pressure_level low
        analyze_results
        ;;
    high)
        run_pressure_level high
        analyze_results
        ;;
    all)
        run_pressure_level low
        sleep 30
        run_pressure_level high
        analyze_results
        ;;
    analyze)
        analyze_results
        ;;
    *)
        echo "未知参数: $1"
        usage
        exit 1
        ;;
esac

log "实验完成 [$(date)]"
