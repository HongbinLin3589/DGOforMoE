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
MAX_LENGTH["bigmath"]=1024

# 按模型大小调整 batch size
# A100 80G × 4 卡：global batch = per_device × accum × 4 = 256
declare -A MODEL_BATCH_SIZE
MODEL_BATCH_SIZE["olmoe"]=32            # 7B:  32×2×4=256  (A100 80G × 4)
MODEL_BATCH_SIZE["olmoe_instruct"]=32   # 7B:  同 olmoe
MODEL_BATCH_SIZE["qwen"]=32            # 14B: 32×2×4=256  (A100 80G × 4)
MODEL_BATCH_SIZE["qwen3"]=8            # 30B: 8×8×4=256   (A100 80G × 4)
MODEL_BATCH_SIZE["qwen3_instruct"]=8   # 30B: 同 qwen3
MODEL_BATCH_SIZE["deepseek"]=16        # 16B: 16×4×4=256  (A100 80G × 4)
MODEL_BATCH_SIZE["mixtral"]=8          # 47B: 8×8×4=256   (A100 80G × 4)

declare -A MODEL_GRAD_ACCUM
MODEL_GRAD_ACCUM["olmoe"]=2
MODEL_GRAD_ACCUM["olmoe_instruct"]=2
MODEL_GRAD_ACCUM["qwen"]=2
MODEL_GRAD_ACCUM["qwen3"]=8
MODEL_GRAD_ACCUM["qwen3_instruct"]=8
MODEL_GRAD_ACCUM["deepseek"]=4
MODEL_GRAD_ACCUM["mixtral"]=8

# 按模型大小选择 deepspeed
declare -A MODEL_DEEPSPEED
MODEL_DEEPSPEED["olmoe"]="zero2"            # 7B  → 单卡可放
MODEL_DEEPSPEED["olmoe_instruct"]="zero2"
MODEL_DEEPSPEED["qwen"]="zero2"             # 14B → 单卡勉强
MODEL_DEEPSPEED["qwen3"]="zero3"            # 30B → 需要分片
MODEL_DEEPSPEED["qwen3_instruct"]="zero3"
MODEL_DEEPSPEED["deepseek"]="zero3"         # 16B
MODEL_DEEPSPEED["mixtral"]="zero3"          # 47B

# LoRA target_modules 配置 - 包含 router gate 以训练路由器
declare -A MODEL_TARGET_MODULES
MODEL_TARGET_MODULES["olmoe"]="gate q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
MODEL_TARGET_MODULES["olmoe_instruct"]="gate q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
MODEL_TARGET_MODULES["qwen"]="gate q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
MODEL_TARGET_MODULES["qwen3"]="gate q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
MODEL_TARGET_MODULES["qwen3_instruct"]="gate q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
MODEL_TARGET_MODULES["deepseek"]="gate q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
MODEL_TARGET_MODULES["mixtral"]="gate q_proj k_proj v_proj o_proj w1 w2 w3"

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

# 获取系统提示（SFT / GRPO / DGO 统一使用 get_system_prompt，no-CoT 格式：仅 <answer>\boxed{}）
SYSTEM_PROMPT=$(get_system_prompt "$DATASET_KEY")

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
    bigmath)
        DATASET_PATH=$(get_dataset_path bigmath)
        # BigMath 与 GSM8K 相同格式: question -> query, answer -> response
        COLUMNS_MAPPING='{"question":"query","answer":"response"}'
        ;;
    *)
        echo "❌ 未知数据集: $DATASET_KEY"
        echo "可用数据集: gsm8k, math, mbpp, bigmath"
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
TARGET_MODULES="${MODEL_TARGET_MODULES[$MODEL_KEY]:-all-linear}"

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
echo "  target_modules: $TARGET_MODULES"
echo "  (注: 包含 router gate，训练路由器以产生路由扰动)"
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
    --template default \
    --template_backend swift \
    --attn_impl sdpa \
    --system "$SYSTEM_PROMPT" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    ${COLUMNS_MAPPING:+--columns "$COLUMNS_MAPPING"} \
    \
    --train_type lora \
    --lora_rank $DEFAULT_LORA_RANK \
    --lora_alpha $DEFAULT_LORA_ALPHA \
    --lora_dropout $DEFAULT_LORA_DROPOUT \
    --target_modules $TARGET_MODULES \
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
    --router_aux_loss_coef 0 \
    --moe_monitor_enabled ${MOE_MONITOR_ENABLED:-$DEFAULT_MOE_MONITOR_ENABLED} \
    --moe_log_every ${MOE_LOG_EVERY:-$DEFAULT_MOE_LOG_EVERY}

echo "✅ SFT 训练完成! 输出目录: $OUTPUT_DIR"
echo "结束时间: $(date)"
