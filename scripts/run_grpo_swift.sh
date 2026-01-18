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
# 环境配置
# =============================================================================
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 8 GPUs

export HF_HOME="/usr/storage/fwan/huggingface_cache"
export HF_HUB_CACHE="/usr/storage/fwan/huggingface_cache/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export USE_HF=1

export NPROC_PER_NODE=8  # 8 GPUs for training
export MASTER_PORT=$((29500 + RANDOM % 100))  # 随机端口避免冲突

source /opt/miniforge3/bin/activate dgo

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR="/usr/commondata/public/hf_hub/cc/DGO"
OUTPUT_BASE="${BASE_DIR}/outputs/swift_grpo"
LOG_DIR="${BASE_DIR}/logs/swift_grpo"
HF_CACHE="/usr/storage/fwan/huggingface_cache/hub"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

# 数据集路径
GSM8K_PATH="gsm8k"
MATH_PATH="EleutherAI/hendrycks_math"
MBPP_DATASET="mbpp"

# =============================================================================
# 模型映射 - 使用本地缓存路径
# =============================================================================
declare -A MODEL_MAP
MODEL_MAP["olmoe"]="${HF_CACHE}/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654"
MODEL_MAP["qwen"]="${HF_CACHE}/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
MODEL_MAP["deepseek"]="${HF_CACHE}/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
MODEL_MAP["mixtral"]="${HF_CACHE}/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0"

# 数据集 max_length 映射
declare -A MAX_LENGTH
MAX_LENGTH["gsm8k"]=1024
MAX_LENGTH["math"]=1024
MAX_LENGTH["mbpp"]=1024

# =============================================================================
# vLLM 配置映射 - 根据模型大小调整
# 参考: ms-swift/examples/train/grpo/internal/moe_lora.sh
# =============================================================================
declare -A VLLM_TP_SIZE
VLLM_TP_SIZE["olmoe"]=1
VLLM_TP_SIZE["qwen"]=1
VLLM_TP_SIZE["deepseek"]=2
VLLM_TP_SIZE["mixtral"]=2

declare -A VLLM_MEM_UTIL
# LoRA训练显存占用小，vLLM可以分配更多用于KV Cache
VLLM_MEM_UTIL["olmoe"]=0.5
VLLM_MEM_UTIL["qwen"]=0.6   # Qwen MoE 较大，需要更多
VLLM_MEM_UTIL["deepseek"]=0.5
VLLM_MEM_UTIL["mixtral"]=0.5

# =============================================================================
# Template 配置映射 - 基础模型使用 default
# =============================================================================
declare -A MODEL_TEMPLATE
MODEL_TEMPLATE["olmoe"]="default"
MODEL_TEMPLATE["qwen"]="default"
MODEL_TEMPLATE["deepseek"]="default"
MODEL_TEMPLATE["mixtral"]="default"

# =============================================================================
# 超参数配置
# =============================================================================
LORA_RANK=8
LORA_ALPHA=64
LORA_DROPOUT=0.05
LEARNING_RATE=5e-6
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
NUM_EPOCHS=5

# 按模型大小调整 batch size (保持 global batch = 256)
declare -A MODEL_BATCH_SIZE
MODEL_BATCH_SIZE["olmoe"]=16      # 小模型，batch=16, grad_accum=2 → 16×2×8=256
MODEL_BATCH_SIZE["qwen"]=16        # 中模型，batch=8, grad_accum=4 → 8×4×8=256
MODEL_BATCH_SIZE["deepseek"]=16    # 中模型，batch=8, grad_accum=4 → 8×4×8=256
MODEL_BATCH_SIZE["mixtral"]=2     # 大模型，batch=2, grad_accum=16 → 2×16×8=256

declare -A MODEL_GRAD_ACCUM
MODEL_GRAD_ACCUM["olmoe"]=2
MODEL_GRAD_ACCUM["qwen"]=2
MODEL_GRAD_ACCUM["deepseek"]=2
MODEL_GRAD_ACCUM["mixtral"]=16

# 按模型大小选择 deepspeed (GRPO 用 zero2 更稳定)
declare -A MODEL_DEEPSPEED
MODEL_DEEPSPEED["olmoe"]="zero2"      # 小模型用 zero2
MODEL_DEEPSPEED["qwen"]="zero2"       # 中模型用 zero2
MODEL_DEEPSPEED["deepseek"]="zero2"   # 中模型用 zero2 (GRPO 需要更稳定)
MODEL_DEEPSPEED["mixtral"]="zero2"    # 大模型也用 zero2 (GRPO 避免 zero3 卡住)

# GRPO 特定参数
NUM_GENERATIONS=8
BETA=0.01

# =============================================================================
# 参数解析
# =============================================================================
MODEL_KEY="${1:-olmoe}"
DATASET_KEY="${2:-gsm8k}"

if [[ -z "${MODEL_MAP[$MODEL_KEY]}" ]]; then
    echo "❌ 未知模型: $MODEL_KEY"
    echo "可用模型: olmoe, qwen, deepseek, mixtral"
    exit 1
fi

MODEL_PATH="${MODEL_MAP[$MODEL_KEY]}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

MAX_LEN="${MAX_LENGTH[$DATASET_KEY]}"

case "$DATASET_KEY" in
    gsm8k)
        DATASET_PATH="$GSM8K_PATH"
        ;;
    math)
        DATASET_PATH="$MATH_PATH"
        ;;
    mbpp)
        DATASET_PATH="$MBPP_DATASET"
        ;;
    *)
        echo "❌ 未知数据集: $DATASET_KEY"
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_KEY}_${DATASET_KEY}"
LOG_FILE="${LOG_DIR}/${MODEL_KEY}_${DATASET_KEY}_$(date +%Y%m%d_%H%M%S).log"

# 立即创建日志文件并开始记录所有输出
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

# =============================================================================
# 打印配置
# =============================================================================
echo "============================================================"
echo "MS-Swift GRPO Training Configuration"
echo "============================================================"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "template: $TEMPLATE"
echo "batch_size: $BATCH_SIZE"
echo "gradient_accumulation: $GRADIENT_ACCUMULATION"
echo "global_batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION * 8))"
echo "deepspeed: $DEEPSPEED"
echo "num_generations: $NUM_GENERATIONS"
echo "max_completion_length: $MAX_LEN"
echo "beta: $BETA"
echo "vllm_tensor_parallel_size: $VLLM_TP"
echo "vllm_gpu_memory_utilization: $VLLM_MEM"
echo "============================================================"

# =============================================================================
# 运行 GRPO 训练
# 参考: ms-swift/examples/train/grpo/internal/moe_lora.sh
# =============================================================================
echo "🚀 开始 GRPO 训练..."

swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --template "$TEMPLATE" \
    --template_backend swift \
    --dataset "$DATASET_PATH" \
    --columns '{"answer": "solution"}' \
    --output_dir "$OUTPUT_DIR" \
    \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization $VLLM_MEM \
    --vllm_tensor_parallel_size $VLLM_TP \
    --vllm_max_model_len $MAX_LEN \
    \
    --train_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
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
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --lr_scheduler_type cosine \
    \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --warmup_ratio $WARMUP_RATIO \
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
    --beta $BETA

echo "✅ GRPO 训练完成! 输出目录: $OUTPUT_DIR"
echo "结束时间: $(date)"
