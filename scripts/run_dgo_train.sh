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
# 环境配置 (和GRPO一致)
# =============================================================================
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HF_HOME="/usr/storage/fwan/huggingface_cache"
export HF_HUB_CACHE="/usr/storage/fwan/huggingface_cache/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export USE_HF=1
export NPROC_PER_NODE=8
export MASTER_PORT=$((29500 + RANDOM % 100))

# 禁用Python输出缓冲，确保日志实时刷新 (适合tmux查看)
export PYTHONUNBUFFERED=1

source /opt/miniforge3/bin/activate dgo

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR="/usr/commondata/public/hf_hub/cc/DGO"
OUTPUT_BASE="${BASE_DIR}/outputs/swift_dgo"
LOG_DIR="${BASE_DIR}/logs/dgo_train"
HF_CACHE="/usr/storage/fwan/huggingface_cache/hub"
DGO_CACHE="${BASE_DIR}/dgo_cache"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

# =============================================================================
# 模型映射 - 和GRPO一致
# =============================================================================
declare -A MODEL_MAP
MODEL_MAP["olmoe"]="${HF_CACHE}/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654"
MODEL_MAP["qwen"]="${HF_CACHE}/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
MODEL_MAP["deepseek"]="${HF_CACHE}/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
MODEL_MAP["mixtral"]="${HF_CACHE}/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0"

# 数据集 max_length 映射 (和GRPO一致)
declare -A MAX_LENGTH
MAX_LENGTH["gsm8k"]=1024
MAX_LENGTH["math"]=1024
MAX_LENGTH["mbpp"]=1024

# =============================================================================
# 超参数配置 - 和GRPO完全一致
# =============================================================================
LORA_RANK=8
LORA_ALPHA=64
LORA_DROPOUT=0.05
LEARNING_RATE=5e-6
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
NUM_EPOCHS=5  # 和GRPO一致

# 按模型大小调整 batch size (保持 global batch = 256, 和GRPO一致)
declare -A MODEL_BATCH_SIZE
MODEL_BATCH_SIZE["olmoe"]=32      # 16×2×8=256
MODEL_BATCH_SIZE["qwen"]=32        # 8×4×8=256
MODEL_BATCH_SIZE["deepseek"]=32    # 8×4×8=256
MODEL_BATCH_SIZE["mixtral"]=32     # 2×16×8=256

declare -A MODEL_GRAD_ACCUM
MODEL_GRAD_ACCUM["olmoe"]=1
MODEL_GRAD_ACCUM["qwen"]=1
MODEL_GRAD_ACCUM["deepseek"]=1
MODEL_GRAD_ACCUM["mixtral"]=1

# DeepSpeed配置 - 和GRPO一致 (用zero2更稳定)
declare -A MODEL_DEEPSPEED
MODEL_DEEPSPEED["olmoe"]="zero2"
MODEL_DEEPSPEED["qwen"]="zero3"
MODEL_DEEPSPEED["deepseek"]="zero3"
MODEL_DEEPSPEED["mixtral"]="zero3"

# =============================================================================
# DGO特定参数
# =============================================================================
# 注意: DGO的beta和GRPO的beta含义不同
# - GRPO beta=0.01: KL散度惩罚系数
# - DGO beta=0.1: 权重温度参数 (w = exp(r/beta) / Z)
#   beta越小，权重分布越尖锐（高reward样本权重更大）
DGO_BETA=0.1

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

# 输出配置
if [[ -n "$FREEZE_ROUTER" ]]; then
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_KEY}_${DATASET_KEY}_frozen"
else
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_KEY}_${DATASET_KEY}"
fi
LOG_FILE="${LOG_DIR}/${MODEL_KEY}_${DATASET_KEY}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "$LOG_FILE")"

# =============================================================================
# 开始记录日志
# =============================================================================
exec > >(tee -a "$LOG_FILE") 2>&1

GLOBAL_BATCH=$((BATCH_SIZE * GRADIENT_ACCUMULATION * 8))

echo "============================================================"
echo "DGO Training (Offline Weighted SFT)"
echo "============================================================"
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
echo "  learning_rate: $LEARNING_RATE"
echo "  deepspeed: $DEEPSPEED"
echo ""
echo "DGO配置:"
echo "  dgo_beta: $DGO_BETA"
echo "  freeze_router: ${FREEZE_ROUTER:-false}"
echo ""
echo "LoRA配置:"
echo "  rank: $LORA_RANK"
echo "  alpha: $LORA_ALPHA"
echo "  dropout: $LORA_DROPOUT"
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
    --reward_funcs accuracy \
    $FREEZE_ROUTER \
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
    --warmup_ratio $WARMUP_RATIO \
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
    --report_to tensorboard

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
