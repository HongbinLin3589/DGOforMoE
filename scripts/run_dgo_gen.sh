#!/bin/bash
# =============================================================================
# DGO Data Generation Script (Inference Only)
# =============================================================================
# 使用方法:
#   bash run_dgo_gen.sh [MODEL_NAME] [DATASET_NAME]
#
# 示例:
#   bash run_dgo_gen.sh olmoe gsm8k
#   bash run_dgo_gen.sh qwen math
#   bash run_dgo_gen.sh deepseek mbpp
#   bash run_dgo_gen.sh mixtral gsm8k
#
# 说明:
#   DGO生成阶段只需要inference，不需要训练
#   使用vLLM进行快速推理，为每个prompt生成N个response
# =============================================================================

set -e

# =============================================================================
# 环境配置
# =============================================================================
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HF_HOME="/usr/storage/fwan/huggingface_cache"
export HF_HUB_CACHE="/usr/storage/fwan/huggingface_cache/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export USE_HF=1

# 禁用Python输出缓冲，确保日志实时刷新 (适合tmux查看)
export PYTHONUNBUFFERED=1

# 禁用FlashInfer sampler (避免JIT编译问题，使用vLLM内置实现)
export VLLM_USE_FLASHINFER_SAMPLER=0

source /opt/miniforge3/bin/activate dgo

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR="/usr/commondata/public/hf_hub/cc/DGO"
OUTPUT_BASE="${BASE_DIR}/dgo_cache"
LOG_DIR="${BASE_DIR}/logs/dgo_gen"
HF_CACHE="/usr/storage/fwan/huggingface_cache/hub"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

# =============================================================================
# 模型映射 - 使用本地缓存路径 (和GRPO一致)
# =============================================================================
declare -A MODEL_MAP
MODEL_MAP["olmoe"]="${HF_CACHE}/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654"
MODEL_MAP["qwen"]="${HF_CACHE}/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
MODEL_MAP["deepseek"]="${HF_CACHE}/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
MODEL_MAP["mixtral"]="${HF_CACHE}/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0"

# =============================================================================
# vLLM 配置映射 - 和GRPO一致
# =============================================================================
declare -A VLLM_TP_SIZE
VLLM_TP_SIZE["olmoe"]=8    # 小模型用8卡加速
VLLM_TP_SIZE["qwen"]=8     # 中模型用8卡加速
VLLM_TP_SIZE["deepseek"]=8 # 中大模型用8卡
VLLM_TP_SIZE["mixtral"]=8  # 大模型(46.7B)需要8卡

declare -A VLLM_MEM_UTIL
VLLM_MEM_UTIL["olmoe"]=0.9
VLLM_MEM_UTIL["qwen"]=0.9
VLLM_MEM_UTIL["deepseek"]=0.9
VLLM_MEM_UTIL["mixtral"]=0.9

# 数据集 max_length 映射 (和GRPO一致)
declare -A MAX_LENGTH
MAX_LENGTH["gsm8k"]=1024
MAX_LENGTH["math"]=1024
MAX_LENGTH["mbpp"]=1024

# =============================================================================
# DGO生成参数 (和GRPO的num_generations一致)
# =============================================================================
NUM_GENERATIONS=8
TEMPERATURE=1.0  # 和GRPO一致
TOP_P=0.95

# =============================================================================
# 批量推理优化参数 (充分利用显存，但避免OOM)
# =============================================================================
# max_num_seqs: 并行处理的最大序列数
#   - 每个prompt生成8个序列，所以实际并行 = max_num_seqs / 8 个prompts
#   - 增大可提高吞吐量，但需要更多显存
#   - 根据Copilot Review降低配置以避免OOM
declare -A MAX_NUM_SEQS
MAX_NUM_SEQS["olmoe"]=256     # 小模型 (1.3B激活)，并行32 prompts
MAX_NUM_SEQS["qwen"]=128      # 中模型 (2.7B激活)，并行16 prompts
MAX_NUM_SEQS["deepseek"]=128  # 中模型 (2.8B激活, TP=2)，并行16 prompts
MAX_NUM_SEQS["mixtral"]=64    # 大模型 (12.9B激活, TP=2)，并行8 prompts

# max_num_batched_tokens: 每批最大token数，更精细的显存控制
declare -A MAX_BATCHED_TOKENS
MAX_BATCHED_TOKENS["olmoe"]=16384     # 小模型可以处理更多tokens
MAX_BATCHED_TOKENS["qwen"]=12288      # 中模型
MAX_BATCHED_TOKENS["deepseek"]=10240  # 较大模型
MAX_BATCHED_TOKENS["mixtral"]=8192    # 大模型

# swap_space: CPU swap空间(GB)，用于支持更大batch
SWAP_SPACE=8

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

VLLM_TP="${VLLM_TP_SIZE[$MODEL_KEY]}"
VLLM_MEM="${VLLM_MEM_UTIL[$MODEL_KEY]}"
MAX_LEN="${MAX_LENGTH[$DATASET_KEY]}"
MAX_SEQS="${MAX_NUM_SEQS[$MODEL_KEY]}"
MAX_TOKENS_BATCH="${MAX_BATCHED_TOKENS[$MODEL_KEY]}"

# 输出配置
OUTPUT_FILE="${OUTPUT_BASE}/dgo_data_${MODEL_KEY}_${DATASET_KEY}.json"
LOG_FILE="${LOG_DIR}/${MODEL_KEY}_${DATASET_KEY}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "$LOG_FILE")"

# =============================================================================
# 开始记录日志
# =============================================================================
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "DGO Data Generation (Inference Only)"
echo "============================================================"
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
# 运行vLLM推理 (只做inference，不训练)
# =============================================================================
echo "🚀 开始DGO数据生成 (vLLM inference)..."

python "${BASE_DIR}/vllm_inference.py" \
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
    --stop $'\n### Human:' --stop '</answer>'

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
