#!/bin/bash
# =============================================================================
# 串行运行所有实验脚本
# =============================================================================
# 使用方法:
#   ./run_all_experiments.sh [sft|grpo|all]
#
# 示例:
#   ./run_all_experiments.sh sft    # 只跑 SFT
#   ./run_all_experiments.sh grpo   # 只跑 GRPO
#   ./run_all_experiments.sh all    # 跑 SFT + GRPO
#   ./run_all_experiments.sh        # 默认跑 all
# =============================================================================

set -e

BASE_DIR="/usr/commondata/public/hf_hub/cc/DGO"
cd "$BASE_DIR"

# 实验配置
MODELS=("olmoe" "qwen" "deepseek" "mixtral")
DATASETS=("gsm8k" "math" "mbpp")

# 解析参数
MODE="${1:-all}"

# 日志文件
MASTER_LOG="${BASE_DIR}/logs/all_experiments_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${BASE_DIR}/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

run_sft() {
    local model=$1
    local dataset=$2
    log "=========================================="
    log "开始 SFT: model=$model, dataset=$dataset"
    log "=========================================="

    if bash scripts/run_sft_swift.sh "$model" "$dataset"; then
        log "✅ SFT 完成: model=$model, dataset=$dataset"
    else
        log "❌ SFT 失败: model=$model, dataset=$dataset"
    fi
}

run_grpo() {
    local model=$1
    local dataset=$2
    log "=========================================="
    log "开始 GRPO: model=$model, dataset=$dataset"
    log "=========================================="

    if bash scripts/run_grpo_swift.sh "$model" "$dataset"; then
        log "✅ GRPO 完成: model=$model, dataset=$dataset"
    else
        log "❌ GRPO 失败: model=$model, dataset=$dataset"
    fi
}

log "=============================================="
log "开始串行实验"
log "模式: $MODE"
log "模型: ${MODELS[*]}"
log "数据集: ${DATASETS[*]}"
log "主日志: $MASTER_LOG"
log "=============================================="

# 计算总实验数
total=0
if [[ "$MODE" == "sft" || "$MODE" == "all" ]]; then
    total=$((total + ${#MODELS[@]} * ${#DATASETS[@]}))
fi
if [[ "$MODE" == "grpo" || "$MODE" == "all" ]]; then
    total=$((total + ${#MODELS[@]} * ${#DATASETS[@]}))
fi

current=0

# 运行 SFT 实验
if [[ "$MODE" == "sft" || "$MODE" == "all" ]]; then
    log ">>> 开始 SFT 实验 <<<"
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            current=$((current + 1))
            log "进度: $current/$total"
            run_sft "$model" "$dataset"
        done
    done
fi

# 运行 GRPO 实验
if [[ "$MODE" == "grpo" || "$MODE" == "all" ]]; then
    log ">>> 开始 GRPO 实验 <<<"
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            current=$((current + 1))
            log "进度: $current/$total"
            run_grpo "$model" "$dataset"
        done
    done
fi

log "=============================================="
log "所有实验完成!"
log "主日志: $MASTER_LOG"
log "=============================================="
