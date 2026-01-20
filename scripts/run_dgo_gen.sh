#!/bin/bash
# =============================================================================
# DGO Data Generation Script (Inference Only)
# =============================================================================
# 使用方法:
#   bash run_dgo_gen.sh [MODEL_NAME] [DATASET_NAME]
#
# 示例:
#   bash run_dgo_gen.sh olmoe gsm8k          # 默认8卡
#   bash run_dgo_gen.sh qwen math
#   bash run_dgo_gen.sh deepseek mbpp
#   bash run_dgo_gen.sh mixtral gsm8k
#
# 指定GPU (4卡示例):
#   GPUS="4,5,6,7" bash run_dgo_gen.sh olmoe math
#
# 调整显存利用率 (有其他进程占用时):
#   GPUS="4,5,6,7" VLLM_MEM=0.5 bash run_dgo_gen.sh olmoe math
#
# 说明:
#   DGO生成阶段只需要inference，不需要训练
#   使用vLLM进行快速推理，为每个prompt生成N个response
# =============================================================================

set -e

# =============================================================================
# 加载环境配置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# 覆盖GPU配置（如果通过GPUS环境变量指定）
if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# 激活 conda 环境
activate_dgo_env

# 禁用Python输出缓冲，确保日志实时刷新
export PYTHONUNBUFFERED=1

# 禁用FlashInfer sampler
export VLLM_USE_FLASHINFER_SAMPLER=0

# 创建必要目录
ensure_dirs

# =============================================================================
# vLLM 配置映射
# =============================================================================
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
DEFAULT_TP_SIZE=${TP_SIZE:-$NUM_GPUS}

declare -A VLLM_TP_SIZE
VLLM_TP_SIZE["olmoe"]=$DEFAULT_TP_SIZE
VLLM_TP_SIZE["qwen"]=$DEFAULT_TP_SIZE
VLLM_TP_SIZE["deepseek"]=$DEFAULT_TP_SIZE
VLLM_TP_SIZE["mixtral"]=$DEFAULT_TP_SIZE

declare -A VLLM_MEM_UTIL
VLLM_MEM_UTIL["olmoe"]=${VLLM_MEM:-0.85}
VLLM_MEM_UTIL["qwen"]=${VLLM_MEM:-0.85}
VLLM_MEM_UTIL["deepseek"]=${VLLM_MEM:-0.85}
VLLM_MEM_UTIL["mixtral"]=${VLLM_MEM:-0.85}

# 数据集 max_length 映射
declare -A MAX_LENGTH
MAX_LENGTH["gsm8k"]=1024
MAX_LENGTH["math"]=1024
MAX_LENGTH["mbpp"]=1024

# =============================================================================
# DGO生成参数
# =============================================================================
NUM_GENERATIONS=8
TEMPERATURE=1.0
TOP_P=0.95

# =============================================================================
# 批量推理优化参数
# =============================================================================
declare -A MAX_NUM_SEQS
MAX_NUM_SEQS["olmoe"]=256
MAX_NUM_SEQS["qwen"]=128
MAX_NUM_SEQS["deepseek"]=128
MAX_NUM_SEQS["mixtral"]=64

declare -A MAX_BATCHED_TOKENS
MAX_BATCHED_TOKENS["olmoe"]=16384
MAX_BATCHED_TOKENS["qwen"]=12288
MAX_BATCHED_TOKENS["deepseek"]=10240
MAX_BATCHED_TOKENS["mixtral"]=8192

SWAP_SPACE=8

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

VLLM_TP="${VLLM_TP_SIZE[$MODEL_KEY]}"
VLLM_MEM="${VLLM_MEM_UTIL[$MODEL_KEY]}"
MAX_LEN="${MAX_LENGTH[$DATASET_KEY]}"
MAX_SEQS="${MAX_NUM_SEQS[$MODEL_KEY]}"
MAX_TOKENS_BATCH="${MAX_BATCHED_TOKENS[$MODEL_KEY]}"

# 输出配置
OUTPUT_FILE="${DGO_CACHE}/dgo_data_${MODEL_KEY}_${DATASET_KEY}.json"
LOG_FILE="${DGO_LOGS_DIR}/../dgo_gen/${MODEL_KEY}_${DATASET_KEY}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "$LOG_FILE")"

# =============================================================================
# 开始记录日志
# =============================================================================
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "DGO Data Generation (Inference Only)"
echo "============================================================"
echo "项目根目录: $DGO_ROOT"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET_KEY"
echo "输出文件: $OUTPUT_FILE"
echo ""
echo "vLLM配置:"
echo "  tensor_parallel_size: $VLLM_TP"
echo "  gpu_memory_utilization: $VLLM_MEM"
echo "  max_tokens: $MAX_LEN"
echo "  max_num_seqs: $MAX_SEQS (并行序列数)"
echo "  max_num_batched_tokens: $MAX_TOKENS_BATCH"
echo "  swap_space: ${SWAP_SPACE}GB"
echo ""
echo "生成配置:"
echo "  num_generations: $NUM_GENERATIONS"
echo "  temperature: $TEMPERATURE"
echo "  top_p: $TOP_P"
echo "  并行prompts: $((MAX_SEQS / NUM_GENERATIONS)) (=$MAX_SEQS / $NUM_GENERATIONS)"
echo "============================================================"
echo "开始时间: $(date)"
echo ""

# =============================================================================
# 运行vLLM推理
# =============================================================================
echo "🚀 开始DGO数据生成 (vLLM inference)..."

python "${DGO_ROOT}/vllm_inference.py" \
    --model_name "$MODEL_PATH" \
    --dataset "$DATASET_KEY" \
    --dataset_split train \
    --n "$NUM_GENERATIONS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_LEN" \
    --tensor_parallel_size "$VLLM_TP" \
    --gpu_memory_utilization "$VLLM_MEM" \
    --max_num_seqs "$MAX_SEQS" \
    --max_num_batched_tokens "$MAX_TOKENS_BATCH" \
    --swap_space "$SWAP_SPACE" \
    --output_file "$OUTPUT_FILE" \
    --stop '</answer>' \
    --stop $'\nQ:' \
    --stop $'\n\nQ:'

echo ""
echo "============================================================"
echo "✅ DGO数据生成完成!"
echo "输出文件: $OUTPUT_FILE"
echo "日志文件: $LOG_FILE"
echo "结束时间: $(date)"
echo "============================================================"

# 检查输出文件
if [ -f "$OUTPUT_FILE" ]; then
    NUM_SAMPLES=$(python -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))")
    echo "✅ 生成 $NUM_SAMPLES 个prompts × $NUM_GENERATIONS generations = $((NUM_SAMPLES * NUM_GENERATIONS)) 个样本"
else
    echo "❌ 输出文件不存在: $OUTPUT_FILE"
    exit 1
fi

echo ""
echo "下一步: 运行DGO训练"
echo "  bash scripts/run_dgo_train.sh $MODEL_KEY $DATASET_KEY"
