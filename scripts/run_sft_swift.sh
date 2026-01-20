#!/bin/bash
# =============================================================================
# MS-Swift SFT Training Script for 4 MoE Models × 3 Datasets
# =============================================================================
# 使用方法:
#   ./run_sft_swift.sh MODEL_NAME DATASET_NAME
#
# 示例:
#   ./run_sft_swift.sh olmoe gsm8k
#   ./run_sft_swift.sh qwen math
#   ./run_sft_swift.sh deepseek mbpp
#   ./run_sft_swift.sh mixtral gsm8k
#
# 模型选项: olmoe, qwen, deepseek, mixtral
# 数据集选项: gsm8k, math, mbpp
# =============================================================================

set -e

# =============================================================================
# 加载环境配置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# 激活 conda 环境
activate_dgo_env

# 创建必要目录
ensure_dirs

# =============================================================================
# 模型特定配置
# =============================================================================
# 数据集 max_length 映射
declare -A MAX_LENGTH
MAX_LENGTH["gsm8k"]=1024
MAX_LENGTH["math"]=1024
MAX_LENGTH["mbpp"]=1024

# 按模型大小调整 batch size (保持 global batch = 256)
declare -A MODEL_BATCH_SIZE
MODEL_BATCH_SIZE["olmoe"]=32      # 小模型，batch=32, grad_accum=1 → 32×1×8=256
MODEL_BATCH_SIZE["qwen"]=8        # 中模型，batch=8, grad_accum=4 → 8×4×8=256
MODEL_BATCH_SIZE["deepseek"]=8    # 中模型，batch=8, grad_accum=4 → 8×4×8=256
MODEL_BATCH_SIZE["mixtral"]=4     # 大模型，batch=4, grad_accum=8 → 4×8×8=256

declare -A MODEL_GRAD_ACCUM
MODEL_GRAD_ACCUM["olmoe"]=1
MODEL_GRAD_ACCUM["qwen"]=4
MODEL_GRAD_ACCUM["deepseek"]=4
MODEL_GRAD_ACCUM["mixtral"]=8

# 按模型大小选择 deepspeed
declare -A MODEL_DEEPSPEED
MODEL_DEEPSPEED["olmoe"]="zero2"      # 小模型用 zero2
MODEL_DEEPSPEED["qwen"]="zero2"       # 中模型用 zero2
MODEL_DEEPSPEED["deepseek"]="zero3"   # 较大模型用 zero3
MODEL_DEEPSPEED["mixtral"]="zero3"    # 大模型用 zero3

# =============================================================================
# 参数解析
# =============================================================================
MODEL_KEY="${1:-olmoe}"
DATASET_KEY="${2:-gsm8k}"

# 验证模型
MODEL_PATH=$(get_model_path "$MODEL_KEY")
if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ 未知模型: $MODEL_KEY"
    echo "可用模型: olmoe, qwen, deepseek, mixtral"
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

MAX_LEN="${MAX_LENGTH[$DATASET_KEY]}"

# 数据集列映射配置
COLUMNS_MAPPING=""

case "$DATASET_KEY" in
    gsm8k)
        DATASET_PATH=$(get_dataset_path gsm8k)
        # GSM8K 数据集列名: question -> query, answer -> response
        COLUMNS_MAPPING='{"question":"query","answer":"response"}'
        ;;
    math)
        DATASET_PATH=$(get_dataset_path math)
        # MATH 数据集列名: problem -> query, solution -> response
        COLUMNS_MAPPING='{"problem":"query","solution":"response"}'
        ;;
    mbpp)
        DATASET_PATH=$(get_dataset_path mbpp)
        # MBPP 数据集列名: problem -> query, solution -> response
        COLUMNS_MAPPING='{"problem":"query","solution":"response"}'
        ;;
    *)
        echo "❌ 未知数据集: $DATASET_KEY"
        echo "可用数据集: gsm8k, math, mbpp"
        exit 1
        ;;
esac

OUTPUT_DIR="${SFT_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}"
LOG_FILE="${SFT_LOGS}/${MODEL_KEY}_${DATASET_KEY}_$(date +%Y%m%d_%H%M%S).log"

# 立即创建日志文件并开始记录所有输出
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"

# =============================================================================
# 获取模型对应的配置
# =============================================================================
BATCH_SIZE="${MODEL_BATCH_SIZE[$MODEL_KEY]:-16}"
GRADIENT_ACCUMULATION="${MODEL_GRAD_ACCUM[$MODEL_KEY]:-2}"
EVAL_BATCH_SIZE=$((BATCH_SIZE * 2))
DEEPSPEED="${MODEL_DEEPSPEED[$MODEL_KEY]:-zero2}"

# =============================================================================
# 打印配置
# =============================================================================
echo "============================================================"
echo "MS-Swift SFT Training Configuration"
echo "============================================================"
echo "项目根目录: $DGO_ROOT"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "batch_size: $BATCH_SIZE"
echo "gradient_accumulation: $GRADIENT_ACCUMULATION"
echo "global_batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION * NPROC_PER_NODE))"
echo "deepspeed: $DEEPSPEED"
echo ""
echo "LoRA 配置:"
echo "  rank: $DEFAULT_LORA_RANK"
echo "  alpha: $DEFAULT_LORA_ALPHA"
echo "  dropout: $DEFAULT_LORA_DROPOUT"
echo ""
echo "MoE 配置:"
echo "  router_aux_loss_coef: ${ROUTER_AUX_LOSS_COEF:-$DEFAULT_ROUTER_AUX_LOSS_COEF}"
echo "  moe_monitor_enabled: ${MOE_MONITOR_ENABLED:-$DEFAULT_MOE_MONITOR_ENABLED}"
echo "  moe_log_every: ${MOE_LOG_EVERY:-$DEFAULT_MOE_LOG_EVERY}"
echo "============================================================"

# =============================================================================
# 运行 SFT 训练
# =============================================================================
echo "🚀 开始训练..."

if [[ -n "$COLUMNS_MAPPING" ]]; then
    echo "  列映射: $COLUMNS_MAPPING"
fi

swift sft \
    --model "$MODEL_PATH" \
    --attn_impl sdpa \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    ${COLUMNS_MAPPING:+--columns "$COLUMNS_MAPPING"} \
    \
    --train_type lora \
    --lora_rank $DEFAULT_LORA_RANK \
    --lora_alpha $DEFAULT_LORA_ALPHA \
    --lora_dropout $DEFAULT_LORA_DROPOUT \
    --target_modules all-linear \
    \
    --learning_rate $DEFAULT_LEARNING_RATE \
    --weight_decay $DEFAULT_WEIGHT_DECAY \
    --warmup_ratio $DEFAULT_WARMUP_RATIO \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --lr_scheduler_type cosine \
    \
    --num_train_epochs $DEFAULT_NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --max_length $MAX_LEN \
    \
    --torch_dtype bfloat16 \
    --gradient_checkpointing true \
    \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 10 \
    \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    \
    --use_hf true \
    --deepspeed $DEEPSPEED \
    \
    --report_to tensorboard \
    \
    --router_aux_loss_coef ${ROUTER_AUX_LOSS_COEF:-$DEFAULT_ROUTER_AUX_LOSS_COEF} \
    --moe_monitor_enabled ${MOE_MONITOR_ENABLED:-$DEFAULT_MOE_MONITOR_ENABLED} \
    --moe_log_every ${MOE_LOG_EVERY:-$DEFAULT_MOE_LOG_EVERY}

echo "✅ SFT 训练完成! 输出目录: $OUTPUT_DIR"
echo "结束时间: $(date)"
