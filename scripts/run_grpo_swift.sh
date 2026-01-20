#!/bin/bash
# =============================================================================
# MS-Swift GRPO Training Script for 4 MoE Models × 3 Datasets
# =============================================================================
# 使用方法:
#   ./run_grpo_swift.sh MODEL_NAME DATASET_NAME
#
# 示例:
#   ./run_grpo_swift.sh olmoe gsm8k
#   ./run_grpo_swift.sh qwen math
#   ./run_grpo_swift.sh deepseek mbpp
#   ./run_grpo_swift.sh mixtral gsm8k
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

# vLLM 配置映射 - 根据模型大小调整
declare -A VLLM_TP_SIZE
VLLM_TP_SIZE["olmoe"]=1
VLLM_TP_SIZE["qwen"]=1
VLLM_TP_SIZE["deepseek"]=2
VLLM_TP_SIZE["mixtral"]=2

declare -A VLLM_MEM_UTIL
VLLM_MEM_UTIL["olmoe"]=0.3
VLLM_MEM_UTIL["qwen"]=0.3
VLLM_MEM_UTIL["deepseek"]=0.3
VLLM_MEM_UTIL["mixtral"]=0.3

# Template 配置映射 - 基础模型使用 default
declare -A MODEL_TEMPLATE
MODEL_TEMPLATE["olmoe"]="default"
MODEL_TEMPLATE["qwen"]="default"
MODEL_TEMPLATE["deepseek"]="default"
MODEL_TEMPLATE["mixtral"]="default"

# 按模型大小调整 batch size
declare -A MODEL_BATCH_SIZE
MODEL_BATCH_SIZE["olmoe"]=8
MODEL_BATCH_SIZE["qwen"]=8
MODEL_BATCH_SIZE["deepseek"]=8
MODEL_BATCH_SIZE["mixtral"]=2

declare -A MODEL_GRAD_ACCUM
MODEL_GRAD_ACCUM["olmoe"]=4
MODEL_GRAD_ACCUM["qwen"]=4
MODEL_GRAD_ACCUM["deepseek"]=4
MODEL_GRAD_ACCUM["mixtral"]=16

# 按模型大小选择 deepspeed (GRPO 用 zero2 更稳定)
declare -A MODEL_DEEPSPEED
MODEL_DEEPSPEED["olmoe"]="zero2"
MODEL_DEEPSPEED["qwen"]="zero2"
MODEL_DEEPSPEED["deepseek"]="zero2"
MODEL_DEEPSPEED["mixtral"]="zero2"

# GRPO 特定参数
declare -A MODEL_NUM_GENERATIONS
MODEL_NUM_GENERATIONS["olmoe"]=4
MODEL_NUM_GENERATIONS["qwen"]=8
MODEL_NUM_GENERATIONS["deepseek"]=8
MODEL_NUM_GENERATIONS["mixtral"]=8

BETA=0.01
NUM_EPOCHS=5

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

# 数据集配置
case "$DATASET_KEY" in
    gsm8k)
        DATASET_PATH=$(get_dataset_path gsm8k)
        REWARD_FUNC="accuracy"
        COLUMNS='{"question": "query", "answer": "solution"}'
        ;;
    math)
        DATASET_PATH=$(get_dataset_path math)
        REWARD_FUNC="accuracy"
        COLUMNS='{"problem": "query"}'
        ;;
    mbpp)
        DATASET_PATH=$(get_dataset_path mbpp)
        REWARD_FUNC="mbpp"
        COLUMNS='{"problem": "query"}'
        ;;
    *)
        echo "❌ 未知数据集: $DATASET_KEY"
        echo "可用数据集: gsm8k, math, mbpp"
        exit 1
        ;;
esac

OUTPUT_DIR="${GRPO_OUTPUT}/${MODEL_KEY}_${DATASET_KEY}"
LOG_FILE="${GRPO_LOGS}/${MODEL_KEY}_${DATASET_KEY}_$(date +%Y%m%d_%H%M%S).log"

# 开始记录日志
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"

# =============================================================================
# 获取模型对应的配置
# =============================================================================
VLLM_TP="${VLLM_TP_SIZE[$MODEL_KEY]:-1}"
VLLM_MEM="${VLLM_MEM_UTIL[$MODEL_KEY]:-0.4}"
TEMPLATE="${MODEL_TEMPLATE[$MODEL_KEY]:-default}"
BATCH_SIZE="${MODEL_BATCH_SIZE[$MODEL_KEY]:-8}"
GRADIENT_ACCUMULATION="${MODEL_GRAD_ACCUM[$MODEL_KEY]:-4}"
DEEPSPEED="${MODEL_DEEPSPEED[$MODEL_KEY]:-zero2}"
NUM_GENERATIONS="${MODEL_NUM_GENERATIONS[$MODEL_KEY]:-8}"

# =============================================================================
# 打印配置
# =============================================================================
echo "============================================================"
echo "MS-Swift GRPO Training Configuration"
echo "============================================================"
echo "项目根目录: $DGO_ROOT"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "template: $TEMPLATE"
echo "columns: $COLUMNS"
echo "batch_size: $BATCH_SIZE"
echo "gradient_accumulation: $GRADIENT_ACCUMULATION"
echo "global_batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION * NPROC_PER_NODE))"
echo "deepspeed: $DEEPSPEED"
echo "num_generations: $NUM_GENERATIONS"
echo "max_completion_length: $MAX_LEN"
echo "beta: $BETA"
echo "reward_func: $REWARD_FUNC"
echo "vllm_tensor_parallel_size: $VLLM_TP"
echo "vllm_gpu_memory_utilization: $VLLM_MEM"
echo ""
echo "MoE 配置:"
echo "  router_aux_loss_coef: ${ROUTER_AUX_LOSS_COEF:-$DEFAULT_ROUTER_AUX_LOSS_COEF}"
echo "  moe_monitor_enabled: ${MOE_MONITOR_ENABLED:-$DEFAULT_MOE_MONITOR_ENABLED}"
echo "  moe_log_every: ${MOE_LOG_EVERY:-$DEFAULT_MOE_LOG_EVERY}"
echo "============================================================"

# =============================================================================
# 运行 GRPO 训练
# =============================================================================
echo "🚀 开始 GRPO 训练..."

swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --template "$TEMPLATE" \
    --template_backend swift \
    --dataset "$DATASET_PATH" \
    --columns "$COLUMNS" \
    --output_dir "$OUTPUT_DIR" \
    \
    --reward_funcs "$REWARD_FUNC" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization $VLLM_MEM \
    --vllm_tensor_parallel_size $VLLM_TP \
    --vllm_max_model_len $MAX_LEN \
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
    \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --warmup_ratio $DEFAULT_WARMUP_RATIO \
    --dataloader_num_workers 4 \
    \
    --num_generations $NUM_GENERATIONS \
    --temperature 1.0 \
    --deepspeed $DEEPSPEED \
    --log_completions true \
    \
    --sleep_level 1 \
    \
    --report_to tensorboard \
    --num_iterations 1 \
    --beta $BETA \
    \
    --router_aux_loss_coef ${ROUTER_AUX_LOSS_COEF:-$DEFAULT_ROUTER_AUX_LOSS_COEF} \
    --moe_monitor_enabled ${MOE_MONITOR_ENABLED:-$DEFAULT_MOE_MONITOR_ENABLED} \
    --moe_log_every ${MOE_LOG_EVERY:-$DEFAULT_MOE_LOG_EVERY}

echo "✅ GRPO 训练完成! 输出目录: $OUTPUT_DIR"
echo "结束时间: $(date)"
