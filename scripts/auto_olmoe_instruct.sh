#!/bin/bash
# =============================================================================
# 全自动 olmoe_instruct 流水线
# 等待 qwen bigmath DGO 训练完成后，自动依次运行：
#   1. SFT
#   2. GRPO (基于 SFT checkpoint-1000)
#   3. DGO gen (基于 SFT checkpoint-1000)
#   4. DGO train (基于 SFT checkpoint-1000)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
activate_dgo_env

OLMOE_GPUS="3,5,6,7"
NPROC=4
MODEL_KEY="olmoe_instruct"
DATASET_KEY="bigmath"

# 同时检查两个可能的日志路径
DGO_TRAIN_LOG_1="/usr/storage/fwan/DGO/logs/swift_dgo/qwen_bigmath_20260403_201418.log"
DGO_TRAIN_LOG_2="/usr/commondata/public/hf_hub/cc/DGOforMoE/logs/swift_dgo/qwen_bigmath_20260403_201418.log"
MONITOR_LOG="${DGO_LOGS_DIR}/../auto_olmoe/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$MONITOR_LOG")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

# 获取当前 DGO 训练 step
get_current_step() {
    for f in "$DGO_TRAIN_LOG_1" "$DGO_TRAIN_LOG_2"; do
        if [ -f "$f" ]; then
            grep -oE "[0-9]+/1252" "$f" 2>/dev/null | tail -1 | cut -d'/' -f1 && return
        fi
    done
    echo "0"
}

# 检查 DGO 训练是否完成
check_training_done() {
    for f in "$DGO_TRAIN_LOG_1" "$DGO_TRAIN_LOG_2"; do
        [ -f "$f" ] || continue
        if grep -q "✅ DGO训练完成" "$f" 2>/dev/null; then return 0; fi
        local step=$(grep -oE "[0-9]+/1252" "$f" 2>/dev/null | tail -1 | cut -d'/' -f1)
        if [ "$step" = "1252" ]; then return 0; fi
    done
    return 1
}

# 检查 GPU 3,5,6,7 是否已释放（<3GB）
check_gpus_free() {
    for gpu in 3 5 6 7; do
        local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=$gpu | sed 's/ MiB//' | tr -d ' ')
        if [ "$mem" -gt 3000 ]; then
            return 1
        fi
    done
    return 0
}

# 带重试的命令执行（最多3次）
run_with_retry() {
    local name="$1"
    local max_retries=3
    shift
    local retry=0
    while [ $retry -lt $max_retries ]; do
        log ">>> 启动 $name (第$((retry+1))次尝试)"
        if "$@"; then
            log "✅ $name 完成"
            return 0
        else
            local exit_code=$?
            log "❌ $name 失败 (exit=$exit_code)，30秒后重试..."
            retry=$((retry+1))
            sleep 30
        fi
    done
    log "❌ $name 连续失败 $max_retries 次，中止流水线"
    return 1
}

# =============================================================================
# Phase 0: 等待 qwen bigmath DGO 训练完成
# =============================================================================
log "============================================================"
log " olmoe_instruct 全自动流水线启动"
log " 监控日志: $MONITOR_LOG"
log " 等待 qwen bigmath DGO 训练完成 (1252 steps)..."
log "============================================================"

while true; do
    STEP=$(get_current_step)
    GPU3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=3 | sed 's/ MiB//' | tr -d ' ')
    GPU5=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=5 | sed 's/ MiB//' | tr -d ' ')
    GPU6=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=6 | sed 's/ MiB//' | tr -d ' ')
    GPU7=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=7 | sed 's/ MiB//' | tr -d ' ')
    log "进度: ${STEP}/1252 | GPU3=${GPU3}MiB GPU5=${GPU5}MiB GPU6=${GPU6}MiB GPU7=${GPU7}MiB"

    if check_training_done && check_gpus_free; then
        log "✅ DGO训练完成 且 GPU已释放，开始 olmoe_instruct 流水线！"
        break
    fi

    sleep 600  # 每10分钟检查一次
done

sleep 30  # 额外等待确保进程完全退出

# =============================================================================
# Phase 1: SFT
# =============================================================================
log ""
log "============================================================"
log " [1/4] olmoe_instruct SFT 训练"
log "============================================================"

run_with_retry "olmoe_instruct SFT" \
    bash -c "cd /usr/commondata/public/hf_hub/cc/DGOforMoE && \
    CUDA_VISIBLE_DEVICES=$OLMOE_GPUS NPROC_PER_NODE=$NPROC \
    bash scripts/run_sft_swift.sh $MODEL_KEY $DATASET_KEY"

# 找 SFT checkpoint-1000
SFT_CKPT=$(ls -d "${SFT_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}"/*/checkpoint-1000 2>/dev/null | tail -1)
if [ -z "$SFT_CKPT" ] || [ ! -d "$SFT_CKPT" ]; then
    log "❌ 找不到 SFT checkpoint-1000，搜索路径: ${SFT_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}"
    exit 1
fi
log "✅ SFT checkpoint-1000: $SFT_CKPT"
sleep 30

# =============================================================================
# Phase 2: GRPO (基于 SFT checkpoint-1000)
# =============================================================================
log ""
log "============================================================"
log " [2/4] olmoe_instruct GRPO 训练 (基于 SFT checkpoint-1000)"
log "  SFT: $SFT_CKPT"
log "============================================================"

run_with_retry "olmoe_instruct GRPO" \
    bash -c "cd /usr/commondata/public/hf_hub/cc/DGOforMoE && \
    CUDA_VISIBLE_DEVICES=$OLMOE_GPUS NPROC_PER_NODE=$NPROC \
    bash scripts/run_grpo_swift.sh $MODEL_KEY $DATASET_KEY '$SFT_CKPT'"

sleep 30

# =============================================================================
# Phase 3: DGO gen (基于 SFT checkpoint-1000, Data Parallel vLLM)
# =============================================================================
log ""
log "============================================================"
log " [3/4] olmoe_instruct DGO gen (DP vLLM, 基于 SFT checkpoint-1000)"
log "  SFT: $SFT_CKPT"
log "============================================================"

run_with_retry "olmoe_instruct DGO gen" \
    bash -c "cd /usr/commondata/public/hf_hub/cc/DGOforMoE && \
    GPUS=$OLMOE_GPUS \
    bash scripts/run_dgo_gen.sh $MODEL_KEY $DATASET_KEY '$SFT_CKPT'"

sleep 30

# =============================================================================
# Phase 4: DGO train (基于 SFT checkpoint-1000)
# =============================================================================
log ""
log "============================================================"
log " [4/4] olmoe_instruct DGO train (基于 SFT checkpoint-1000)"
log "  SFT: $SFT_CKPT"
log "============================================================"

run_with_retry "olmoe_instruct DGO train" \
    bash -c "cd /usr/commondata/public/hf_hub/cc/DGOforMoE && \
    CUDA_VISIBLE_DEVICES=$OLMOE_GPUS NPROC_PER_NODE=$NPROC \
    bash scripts/run_dgo_train.sh $MODEL_KEY $DATASET_KEY '$SFT_CKPT'"

# =============================================================================
# 完成
# =============================================================================
log ""
log "============================================================"
log " 🎉 olmoe_instruct 全流水线完成!"
log "   SFT:       ${SFT_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}/"
log "   GRPO:      ${GRPO_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}/"
log "   DGO data:  ${DGO_CACHE}/dgo_data_${MODEL_KEY}_${DATASET_KEY}.json"
log "   DGO train: ${DGO_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}/"
log " 监控日志: $MONITOR_LOG"
log "============================================================"
