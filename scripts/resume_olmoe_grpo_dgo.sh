#!/bin/bash
# Resume olmoe_instruct pipeline from GRPO (SFT already done)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
activate_dgo_env

OLMOE_GPUS="3,5,6,7"
NPROC=4
MODEL_KEY="olmoe_instruct"
DATASET_KEY="bigmath"
SFT_CKPT="/usr/storage/fwan/DGO/outputs/swift_sft/olmoe_instruct_bigmath/v0-20260404-050152/checkpoint-1000"

RESUME_LOG="${DGO_LOGS_DIR}/../auto_olmoe/resume_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$RESUME_LOG")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESUME_LOG"
}

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
            log "❌ $name 失败，30秒后重试..."
            retry=$((retry+1))
            sleep 30
        fi
    done
    log "❌ $name 连续失败 $max_retries 次，中止"
    return 1
}

log "============================================================"
log " olmoe_instruct 续跑 (从GRPO开始)"
log " SFT checkpoint: $SFT_CKPT"
log " 日志: $RESUME_LOG"
log "============================================================"

# 验证SFT checkpoint
if [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
    log "❌ SFT checkpoint 不存在: $SFT_CKPT"
    exit 1
fi

# =============================================================================
# Phase 2: GRPO
# =============================================================================
log ""
log " [2/4] olmoe_instruct GRPO (基于 SFT checkpoint-1000)"
run_with_retry "olmoe_instruct GRPO" \
    bash -c "cd /usr/commondata/public/hf_hub/cc/DGOforMoE && \
    CUDA_VISIBLE_DEVICES=$OLMOE_GPUS NPROC_PER_NODE=$NPROC \
    bash scripts/run_grpo_swift.sh $MODEL_KEY $DATASET_KEY '$SFT_CKPT'"
sleep 30

# =============================================================================
# Phase 3: DGO gen
# =============================================================================
log ""
log " [3/4] olmoe_instruct DGO gen (基于 SFT checkpoint-1000)"
run_with_retry "olmoe_instruct DGO gen" \
    bash -c "cd /usr/commondata/public/hf_hub/cc/DGOforMoE && \
    GPUS=$OLMOE_GPUS \
    bash scripts/run_dgo_gen.sh $MODEL_KEY $DATASET_KEY '$SFT_CKPT'"
sleep 30

# =============================================================================
# Phase 4: DGO train
# =============================================================================
log ""
log " [4/4] olmoe_instruct DGO train (基于 SFT checkpoint-1000)"
run_with_retry "olmoe_instruct DGO train" \
    bash -c "cd /usr/commondata/public/hf_hub/cc/DGOforMoE && \
    CUDA_VISIBLE_DEVICES=$OLMOE_GPUS NPROC_PER_NODE=$NPROC \
    bash scripts/run_dgo_train.sh $MODEL_KEY $DATASET_KEY '$SFT_CKPT'"

log ""
log "============================================================"
log " 🎉 olmoe_instruct 全流水线完成!"
log "  GRPO:      ${GRPO_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}/"
log "  DGO data:  ${DGO_CACHE}/dgo_data_${MODEL_KEY}_${DATASET_KEY}.json"
log "  DGO train: ${DGO_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}/"
log "============================================================"
