#!/bin/bash
# =============================================================================
# DGO Training Script (Offline Weighted SFT)
# =============================================================================
# 使用方法:
#   bash run_dgo_train.sh [MODEL_NAME] [DATASET_NAME] [--freeze_router]
#
# 示例:
#   bash run_dgo_train.sh olmoe gsm8k              # Group C: 可训练router
#   bash run_dgo_train.sh olmoe gsm8k --freeze_router  # Group D: 冻结router
#   bash run_dgo_train.sh qwen math
#   bash run_dgo_train.sh deepseek mbpp
#   bash run_dgo_train.sh mixtral gsm8k
#
# 说明:
#   DGO训练阶段是offline weighted SFT
#   使用预生成的数据进行训练，每个样本有预计算的权重
#   参数和GRPO保持一致 (epochs=5, global_batch=256, 等)
# =============================================================================

set -e

# =============================================================================
# 加载环境配置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# 激活 conda 环境
activate_dgo_env

# 禁用Python输出缓冲
export PYTHONUNBUFFERED=1

# 允许执行代码评估 (MBPP需要)
export HF_ALLOW_CODE_EVAL=1

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

# 按模型大小调整 batch size
declare -A MODEL_BATCH_SIZE
MODEL_BATCH_SIZE["olmoe"]=32
MODEL_BATCH_SIZE["qwen"]=32
MODEL_BATCH_SIZE["deepseek"]=32
MODEL_BATCH_SIZE["mixtral"]=32

declare -A MODEL_GRAD_ACCUM
MODEL_GRAD_ACCUM["olmoe"]=1
MODEL_GRAD_ACCUM["qwen"]=1
MODEL_GRAD_ACCUM["deepseek"]=1
MODEL_GRAD_ACCUM["mixtral"]=1

# DeepSpeed配置
declare -A MODEL_DEEPSPEED
MODEL_DEEPSPEED["olmoe"]="zero2"
MODEL_DEEPSPEED["qwen"]="zero3"
MODEL_DEEPSPEED["deepseek"]="zero3"
MODEL_DEEPSPEED["mixtral"]="zero3"

# =============================================================================
# DGO特定参数
# =============================================================================
DGO_BETA=0.1
NUM_EPOCHS=5

# =============================================================================
# 参数解析
# =============================================================================
MODEL_KEY="${1:-olmoe}"
DATASET_KEY="${2:-gsm8k}"
FREEZE_ROUTER=""

# 检查是否有--freeze_router参数
for arg in "$@"; do
    if [[ "$arg" == "--freeze_router" ]]; then
        FREEZE_ROUTER="--freeze_router true"
        echo "📌 启用router冻结 (Group D实验)"
    fi
done

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

# DGO数据文件
DGO_DATA_FILE="${DGO_CACHE}/dgo_data_${MODEL_KEY}_${DATASET_KEY}.json"

if [[ ! -f "$DGO_DATA_FILE" ]]; then
    echo "❌ DGO数据文件不存在: $DGO_DATA_FILE"
    echo "请先运行: bash scripts/run_dgo_gen.sh $MODEL_KEY $DATASET_KEY"
    exit 1
fi

# 获取模型配置
MAX_LEN="${MAX_LENGTH[$DATASET_KEY]}"
BATCH_SIZE="${MODEL_BATCH_SIZE[$MODEL_KEY]}"
GRADIENT_ACCUMULATION="${MODEL_GRAD_ACCUM[$MODEL_KEY]}"
DEEPSPEED="${MODEL_DEEPSPEED[$MODEL_KEY]}"

# 根据数据集选择正确的 reward 函数
if [[ "$DATASET_KEY" == "mbpp" ]]; then
    REWARD_FUNC="mbpp"
else
    REWARD_FUNC="accuracy"
fi

# 输出配置
if [[ -n "$FREEZE_ROUTER" ]]; then
    OUTPUT_DIR="${DGO_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}_frozen"
else
    OUTPUT_DIR="${DGO_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}"
fi
LOG_FILE="${DGO_LOGS_DIR}/${MODEL_KEY}_${DATASET_KEY}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "$LOG_FILE")"

# =============================================================================
# 开始记录日志
# =============================================================================
exec > >(tee -a "$LOG_FILE") 2>&1

GLOBAL_BATCH=$((BATCH_SIZE * GRADIENT_ACCUMULATION * NPROC_PER_NODE))

echo "============================================================"
echo "DGO Training (Offline Weighted SFT)"
echo "============================================================"
echo "项目根目录: $DGO_ROOT"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET_KEY"
echo "DGO数据: $DGO_DATA_FILE"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "训练配置 (和GRPO一致):"
echo "  num_epochs: $NUM_EPOCHS"
echo "  per_device_batch_size: $BATCH_SIZE"
echo "  gradient_accumulation: $GRADIENT_ACCUMULATION"
echo "  global_batch_size: $GLOBAL_BATCH"
echo "  learning_rate: $DEFAULT_LEARNING_RATE"
echo "  deepspeed: $DEEPSPEED"
echo ""
echo "DGO配置:"
echo "  dgo_beta: $DGO_BETA"
echo "  freeze_router: ${FREEZE_ROUTER:-false}"
echo "  reward_func: $REWARD_FUNC"
echo ""
echo "LoRA配置:"
echo "  rank: $DEFAULT_LORA_RANK"
echo "  alpha: $DEFAULT_LORA_ALPHA"
echo "  dropout: $DEFAULT_LORA_DROPOUT"
echo ""
echo "MoE 配置:"
echo "  router_aux_loss_coef: ${ROUTER_AUX_LOSS_COEF:-$DEFAULT_ROUTER_AUX_LOSS_COEF}"
echo "  moe_monitor_enabled: ${MOE_MONITOR_ENABLED:-$DEFAULT_MOE_MONITOR_ENABLED}"
echo "  moe_log_every: ${MOE_LOG_EVERY:-$DEFAULT_MOE_LOG_EVERY}"
echo "============================================================"
echo "开始时间: $(date)"
echo ""

# =============================================================================
# 运行DGO训练
# =============================================================================
echo "🚀 开始DGO训练..."

swift rlhf \
    --rlhf_type dgo \
    --model "$MODEL_PATH" \
    --template default \
    --template_backend swift \
    --output_dir "$OUTPUT_DIR" \
    \
    --dgo_data_file "$DGO_DATA_FILE" \
    --dgo_beta "$DGO_BETA" \
    --reward_funcs "$REWARD_FUNC" \
    $FREEZE_ROUTER \
    \
    --train_type lora \
    --lora_rank $DEFAULT_LORA_RANK \
    --lora_alpha $DEFAULT_LORA_ALPHA \
    --lora_dropout $DEFAULT_LORA_DROPOUT \
    --target_modules all-linear \
    \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --max_length $MAX_LEN \
    --max_completion_length $MAX_LEN \
    --overlong_filter true \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $DEFAULT_LEARNING_RATE \
    --weight_decay $DEFAULT_WEIGHT_DECAY \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --lr_scheduler_type cosine \
    --warmup_ratio $DEFAULT_WARMUP_RATIO \
    \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    \
    --deepspeed $DEEPSPEED \
    --report_to tensorboard \
    \
    --router_aux_loss_coef ${ROUTER_AUX_LOSS_COEF:-$DEFAULT_ROUTER_AUX_LOSS_COEF} \
    --moe_monitor_enabled ${MOE_MONITOR_ENABLED:-$DEFAULT_MOE_MONITOR_ENABLED} \
    --moe_log_every ${MOE_LOG_EVERY:-$DEFAULT_MOE_LOG_EVERY}

echo ""
echo "============================================================"
echo "✅ DGO训练完成!"
echo "模型保存到: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "结束时间: $(date)"
echo "============================================================"

# 检查输出目录
if [ -d "$OUTPUT_DIR" ]; then
    NUM_CHECKPOINTS=$(find "$OUTPUT_DIR" -maxdepth 1 -name "checkpoint-*" 2>/dev/null | wc -l)
    echo "✅ 保存了 $NUM_CHECKPOINTS 个checkpoints"

    LAST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | tail -1)
    if [ -n "$LAST_CKPT" ]; then
        echo "📁 最新checkpoint: $LAST_CKPT"
    fi
else
    echo "⚠️ 输出目录未找到: $OUTPUT_DIR"
fi

echo ""
echo "下一步:"
echo "1. 评估模型: lm-eval run --model hf --model_args pretrained=$OUTPUT_DIR --tasks gsm8k"
echo "2. 合并LoRA: swift export --ckpt_dir $OUTPUT_DIR --merge_lora true"
