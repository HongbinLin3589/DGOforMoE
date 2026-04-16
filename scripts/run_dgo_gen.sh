#!/bin/bash
# =============================================================================
# DGO Data Generation Script (Inference Only)
# =============================================================================
# 使用方法:
#   bash run_dgo_gen.sh [MODEL_NAME] [DATASET_NAME] [SFT_CHECKPOINT]
#
# 示例:
#   bash run_dgo_gen.sh olmoe gsm8k                           # 从基础模型开始
#   bash run_dgo_gen.sh olmoe gsm8k /path/to/sft/checkpoint   # 从SFT checkpoint继续
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
#   SFT_CHECKPOINT: 可选，SFT训练后的LoRA checkpoint路径
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

# 使用 vLLM v1 engine（默认）；enforce_eager=True 已在 vllm_inference.py 中禁用 CUDA graph
# 注：VLLM_USE_V1=0 在 vLLM 0.11.0 中会导致路由冲突（V1 engine 检测到 envs.VLLM_USE_V1=False 就崩溃）
export VLLM_USE_V1=1

# 禁用 custom all_reduce（GPU 间无 NVLink/P2P 时会报 invalid argument）
export VLLM_USE_CUSTOM_ALL_REDUCE=0

# 修复 libstdc++ 版本问题（系统库太旧，用 conda 环境里的新版本）
export LD_PRELOAD="/opt/miniforge3/envs/DGO/lib/libstdc++.so.6:${LD_PRELOAD}"

# 创建必要目录
ensure_dirs

# =============================================================================
# vLLM 配置映射
# =============================================================================
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
DEFAULT_TP_SIZE=${TP_SIZE:-$NUM_GPUS}

declare -A VLLM_TP_SIZE
VLLM_TP_SIZE["olmoe"]=$DEFAULT_TP_SIZE
VLLM_TP_SIZE["olmoe_instruct"]=$DEFAULT_TP_SIZE
VLLM_TP_SIZE["qwen"]=$DEFAULT_TP_SIZE
VLLM_TP_SIZE["qwen3"]=$DEFAULT_TP_SIZE
VLLM_TP_SIZE["qwen3_instruct"]=$DEFAULT_TP_SIZE
VLLM_TP_SIZE["deepseek"]=$DEFAULT_TP_SIZE
VLLM_TP_SIZE["mixtral"]=$DEFAULT_TP_SIZE

declare -A VLLM_MEM_UTIL
VLLM_MEM_UTIL["olmoe"]=${VLLM_MEM:-0.85}
VLLM_MEM_UTIL["olmoe_instruct"]=${VLLM_MEM:-0.85}
VLLM_MEM_UTIL["qwen"]=${VLLM_MEM:-0.85}
VLLM_MEM_UTIL["qwen3"]=${VLLM_MEM:-0.85}
VLLM_MEM_UTIL["qwen3_instruct"]=${VLLM_MEM:-0.85}
VLLM_MEM_UTIL["deepseek"]=${VLLM_MEM:-0.85}
VLLM_MEM_UTIL["mixtral"]=${VLLM_MEM:-0.85}

# 数据集 max_length 映射
declare -A MAX_LENGTH
MAX_LENGTH["gsm8k"]=1024
MAX_LENGTH["math"]=1024
MAX_LENGTH["mbpp"]=1024
MAX_LENGTH["bigmath"]=1024

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
MAX_NUM_SEQS["olmoe"]=512
MAX_NUM_SEQS["olmoe_instruct"]=512
MAX_NUM_SEQS["qwen"]=128
MAX_NUM_SEQS["qwen3"]=64
MAX_NUM_SEQS["qwen3_instruct"]=64
MAX_NUM_SEQS["deepseek"]=128
MAX_NUM_SEQS["mixtral"]=64

declare -A MAX_BATCHED_TOKENS
MAX_BATCHED_TOKENS["olmoe"]=32768
MAX_BATCHED_TOKENS["olmoe_instruct"]=32768
MAX_BATCHED_TOKENS["qwen"]=12288
MAX_BATCHED_TOKENS["qwen3"]=8192
MAX_BATCHED_TOKENS["qwen3_instruct"]=8192
MAX_BATCHED_TOKENS["deepseek"]=10240
MAX_BATCHED_TOKENS["mixtral"]=8192

SWAP_SPACE=8

# =============================================================================
# 参数解析
# =============================================================================
MODEL_KEY="${1:-olmoe}"
DATASET_KEY="${2:-gsm8k}"
SFT_CHECKPOINT="${3:-}"  # 可选：SFT checkpoint路径

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

# 验证数据集 key
case "$DATASET_KEY" in
    gsm8k|math|mbpp|bigmath) ;;
    *)
        echo "❌ 未知数据集: $DATASET_KEY"
        echo "可用数据集: gsm8k, math, mbpp, bigmath"
        exit 1
        ;;
esac

VLLM_TP="${VLLM_TP_SIZE[$MODEL_KEY]}"
VLLM_MEM="${VLLM_MEM_UTIL[$MODEL_KEY]}"
MAX_LEN="${MAX_LENGTH[$DATASET_KEY]}"
MAX_SEQS="${MAX_NUM_SEQS[$MODEL_KEY]}"
MAX_TOKENS_BATCH="${MAX_BATCHED_TOKENS[$MODEL_KEY]}"

# 从 env.sh 读取统一的 system prompt（SFT/GRPO/DGO/eval/gen 同源），避免与 vllm_inference.py 内的默认值漂移
SYSTEM_PROMPT=$(get_system_prompt "$DATASET_KEY")

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
echo ""
echo "SFT Checkpoint:"
if [[ -n "$SFT_CHECKPOINT" ]]; then
    echo "  lora_path: $SFT_CHECKPOINT"
else
    echo "  lora_path: (从基础模型开始)"
fi
echo "============================================================"
echo "开始时间: $(date)"
echo ""

# =============================================================================
# 运行vLLM推理
# =============================================================================
echo "🚀 开始DGO数据生成 (vLLM inference)..."

# 处理 SFT checkpoint：vLLM 不支持 MoE 模型的 LoRA，需要先合并
LORA_ARG=""
INFERENCE_MODEL="$MODEL_PATH"
if [[ -n "$SFT_CHECKPOINT" ]]; then
    if [[ ! -d "$SFT_CHECKPOINT" ]]; then
        echo "❌ SFT checkpoint 路径不存在: $SFT_CHECKPOINT"
        exit 1
    fi

    # vLLM 不支持 OLMoE/MoE 模型的 LoRA，需要先合并 adapter 到 base model
    MERGED_DIR="${SFT_CHECKPOINT}-merged"
    if [[ -d "$MERGED_DIR" && -f "$MERGED_DIR/config.json" ]]; then
        echo "📦 发现已合并的模型: $MERGED_DIR (跳过合并)"
    else
        echo "📦 合并 SFT LoRA adapter 到 base model..."
        echo "   checkpoint: $SFT_CHECKPOINT"
        echo "   输出目录: $MERGED_DIR"
        swift export \
            --ckpt_dir "$SFT_CHECKPOINT" \
            --merge_lora true \
            --output_dir "$MERGED_DIR"
        if [[ ! -d "$MERGED_DIR" ]]; then
            echo "❌ 合并失败，输出目录不存在: $MERGED_DIR"
            exit 1
        fi
        echo "✅ LoRA 合并完成"
    fi
    INFERENCE_MODEL="$MERGED_DIR"
    echo "📦 使用合并后的模型: $INFERENCE_MODEL"
fi

# 判断是否使用 Data Parallel（TP=1 多进程）还是 Tensor Parallel
# 小模型（单卡可放下）用 DP 更快：每张卡独立跑模型，处理不同数据
USE_DATA_PARALLEL=${USE_DATA_PARALLEL:-"auto"}
if [[ "$USE_DATA_PARALLEL" == "auto" ]]; then
    # OLMoE 等小模型（<15B）用 DP，大模型用 TP
    case "$MODEL_KEY" in
        olmoe|olmoe_instruct) USE_DATA_PARALLEL="true" ;;   # OLMoE 7B → 单卡可放，用 DP
        *)                    USE_DATA_PARALLEL="false" ;;  # 其他模型 → 需要 TP
    esac
fi

if [[ "$USE_DATA_PARALLEL" == "true" && "$NUM_GPUS" -gt 1 ]]; then
    echo ">>> 使用 Data Parallel 模式: ${NUM_GPUS} 个独立 vLLM 进程 (TP=1)"

    # 获取 GPU 列表
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
    PIDS=()

    for i in $(seq 0 $((NUM_GPUS - 1))); do
        SHARD_FILE="${OUTPUT_FILE%.json}_shard${i}.json"
        echo "  启动 shard $i/${NUM_GPUS} on GPU ${GPU_LIST[$i]} → $SHARD_FILE"

        CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" python "${DGO_ROOT}/vllm_inference.py" \
            --model_name "$INFERENCE_MODEL" \
            --dataset "$DATASET_KEY" \
            --dataset_split train \
            --n "$NUM_GENERATIONS" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --max_tokens "$MAX_LEN" \
            --tensor_parallel_size 1 \
            --gpu_memory_utilization "$VLLM_MEM" \
            --max_num_seqs "$MAX_SEQS" \
            --max_num_batched_tokens "$MAX_TOKENS_BATCH" \
            --swap_space "$SWAP_SPACE" \
            --output_file "$SHARD_FILE" \
            --shard_id "$i" \
            --num_shards "$NUM_GPUS" \
            --system_prompt "$SYSTEM_PROMPT" \
            --stop '</answer>' \
            --stop $'\nQ:' \
            --stop $'\n\nQ:' &

        PIDS+=($!)
    done

    # 等待所有进程完成（轮询显示进度，每张卡数据量可能不均匀）
    echo ">>> 等待 ${NUM_GPUS} 个推理进程完成..."
    START_WAIT=$(date +%s)
    FAIL=0
    while true; do
        RUNNING=0
        DONE_LIST=""
        RUNNING_LIST=""
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                RUNNING=$((RUNNING + 1))
                RUNNING_LIST="${RUNNING_LIST} shard${i}(GPU${GPU_LIST[$i]})"
            else
                DONE_LIST="${DONE_LIST} shard${i}"
            fi
        done

        if [[ "$RUNNING" -eq 0 ]]; then
            break
        fi

        ELAPSED=$(( $(date +%s) - START_WAIT ))
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))
        echo "  [${ELAPSED_MIN}m${ELAPSED_SEC}s] 完成: $((NUM_GPUS - RUNNING))/${NUM_GPUS} | 仍在运行:${RUNNING_LIST}"
        sleep 30
    done

    # 检查各进程退出状态
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "❌ shard${i} (PID ${PIDS[$i]}) 失败"
            FAIL=1
        fi
    done

    TOTAL_WAIT=$(( $(date +%s) - START_WAIT ))
    echo ">>> 所有进程完成，耗时 $((TOTAL_WAIT / 60))m$((TOTAL_WAIT % 60))s"

    if [[ "$FAIL" -eq 1 ]]; then
        echo "❌ 部分推理进程失败，退出"
        exit 1
    fi

    # 合并所有分片结果
    echo ">>> 合并 ${NUM_GPUS} 个分片..."
    python -c "
import json, glob, sys
shards = sorted(glob.glob('${OUTPUT_FILE%.json}_shard*.json'))
merged = []
for s in shards:
    with open(s) as f:
        merged.extend(json.load(f))
with open('${OUTPUT_FILE}', 'w') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print(f'合并完成: {len(shards)} 个分片 → {len(merged)} 条数据')
# 清理分片文件
import os
for s in shards:
    os.remove(s)
"

else
    echo ">>> 使用 Tensor Parallel 模式: TP=${VLLM_TP}"
    python "${DGO_ROOT}/vllm_inference.py" \
        --model_name "$INFERENCE_MODEL" \
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
        --system_prompt "$SYSTEM_PROMPT" \
        --stop '</answer>' \
        --stop $'\nQ:' \
        --stop $'\n\nQ:'
fi

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
