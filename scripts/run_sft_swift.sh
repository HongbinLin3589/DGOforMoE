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
# 环境配置
# =============================================================================
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 8 GPUs

export HF_HOME="/usr/storage/fwan/huggingface_cache"
export HF_HUB_CACHE="/usr/storage/fwan/huggingface_cache/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export USE_HF=1

export NPROC_PER_NODE=8  # 必须与 CUDA_VISIBLE_DEVICES 数量一致
export MASTER_PORT=$((29500 + RANDOM % 100))  # 随机端口避免冲突

source /opt/miniforge3/bin/activate dgo

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR="/usr/commondata/public/hf_hub/cc/DGO"
OUTPUT_BASE="${BASE_DIR}/outputs/swift_sft"
LOG_DIR="${BASE_DIR}/logs/swift_sft"
HF_CACHE="/usr/storage/fwan/huggingface_cache/hub"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

# 数据集路径
GSM8K_PATH="gsm8k"
MATH_PATH="${BASE_DIR}/ReDit/dataset/MATH/data/train/0000.parquet"
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
NUM_EPOCHS=35

# 按模型大小调整 batch size (保持 global batch = 256)
declare -A MODEL_BATCH_SIZE
MODEL_BATCH_SIZE["olmoe"]=16      # 小模型，batch=32, grad_accum=1 → 32×1×8=256
MODEL_BATCH_SIZE["qwen"]=8       # 中模型，batch=16, grad_accum=2 → 16×2×8=256
MODEL_BATCH_SIZE["deepseek"]=8   # 中模型，batch=16, grad_accum=2 → 16×2×8=256
MODEL_BATCH_SIZE["mixtral"]=4    # 大模型，batch=8, grad_accum=4 → 8×4×8=256

declare -A MODEL_GRAD_ACCUM
MODEL_GRAD_ACCUM["olmoe"]=2
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
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "batch_size: $BATCH_SIZE"
echo "gradient_accumulation: $GRADIENT_ACCUMULATION"
echo "global_batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION * 8))"
echo "deepspeed: $DEEPSPEED"
echo "============================================================"

# =============================================================================
# 运行 SFT 训练 (按照 README 格式)
# =============================================================================
echo "🚀 开始训练..."

swift sft \
    --model "$MODEL_PATH" \
    --attn_impl sdpa \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    \
    --train_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --target_modules all-linear \
    \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --lr_scheduler_type cosine \
    \
    --num_train_epochs $NUM_EPOCHS \
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
    --report_to tensorboard

echo "✅ SFT 训练完成! 输出目录: $OUTPUT_DIR"
echo "结束时间: $(date)"
