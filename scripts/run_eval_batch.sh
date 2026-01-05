#!/bin/bash
# =============================================================================
# Batch Evaluation Script - 批量评估多个模型
# =============================================================================
# 使用方法:
#   bash run_eval_batch.sh [task] [batch_size]
#
# 示例:
#   bash run_eval_batch.sh gsm8k_cot 4    # 评估所有模型在gsm8k_cot
#   bash run_eval_batch.sh gsm8k          # 使用gsm8k任务
#   bash run_eval_batch.sh                # 默认: gsm8k_cot, batch_size=auto
#
# 说明:
#   自动评估 SFT, GRPO, DGO 模型，生成对比报告
# =============================================================================

set -euo pipefail

# =============================================================================
# 环境配置 (和训练脚本一致)
# =============================================================================
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HF_HOME="/usr/storage/fwan/huggingface_cache"
export HF_HUB_CACHE="/usr/storage/fwan/huggingface_cache/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# GPU数量 (根据CUDA_VISIBLE_DEVICES自动计算)
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

source /opt/miniforge3/bin/activate dgo 2>/dev/null || true

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR="/usr/commondata/public/hf_hub/cc/DGO"
OUTPUT_DIR="${BASE_DIR}/eval_results"
LOG_DIR="${BASE_DIR}/logs/eval"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# =============================================================================
# 参数解析
# =============================================================================
TASK="${1:-gsm8k_cot}"
BATCH_SIZE="${2:-auto}"

# Batch Size自动调整
if [[ "$BATCH_SIZE" == "auto" ]]; then
    # OLMoE-1B-7B 在 80GB GPU 上可以使用更大的 batch_size
    if [[ "$NUM_GPUS" -ge 8 ]]; then
        BATCH_SIZE=8  # 80GB GPU
    elif [[ "$NUM_GPUS" -ge 4 ]]; then
        BATCH_SIZE=8
    else
        BATCH_SIZE=8
    fi
fi

# =============================================================================
# 定义要评估的模型
# =============================================================================
declare -A MODELS

# OLMoE GSM8K 模型
MODELS["sft_olmoe_gsm8k"]="${BASE_DIR}/outputs/swift_sft/olmoe_gsm8k/v0-20260104-050951/checkpoint-1000"
MODELS["grpo_olmoe_gsm8k"]="${BASE_DIR}/outputs/swift_grpo/olmoe_gsm8k/v0-20260104-073639/checkpoint-1000"

# DGO模型 (如果存在，自动查找最新checkpoint)
if [[ -d "${BASE_DIR}/outputs/swift_dgo/olmoe_gsm8k" ]]; then
    LATEST_DGO=$(find "${BASE_DIR}/outputs/swift_dgo/olmoe_gsm8k" -maxdepth 2 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [[ -n "$LATEST_DGO" ]]; then
        MODELS["dgo_olmoe_gsm8k"]="$LATEST_DGO"
    fi
fi

# =============================================================================
# 日志文件设置
# =============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${OUTPUT_DIR}/batch_summary_${TIMESTAMP}.txt"
LOG_FILE="${LOG_DIR}/batch_eval_${TIMESTAMP}.log"

# =============================================================================
# 开始记录日志 (和训练脚本一致)
# =============================================================================
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"

# =============================================================================
# GPU清理函数
# =============================================================================
cleanup_gpu() {
    echo "清理GPU缓存..."
    python3 << 'PYTHON_EOF'
import torch
import gc
try:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU缓存已清理")
except Exception as e:
    print(f"GPU清理警告: {e}")
PYTHON_EOF
    sleep 3
}

# =============================================================================
# 结果提取函数 (改进版)
# =============================================================================
extract_results() {
    local OUTPUT_FILE=$1

    python3 << PYTHON_EOF
import json
import sys

output_file = "$OUTPUT_FILE"

try:
    with open(output_file) as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results:
        print("N/A (no results)")
        sys.exit(0)

    # 按优先级尝试不同指标
    metric_priority = [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match",
        "acc,none",
        "acc_norm,none",
        "acc"
    ]

    all_results = []
    for task, metrics in results.items():
        for metric_name in metric_priority:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)):
                    # 假设[0,1]范围的是比例，需要转为百分比
                    if 0 <= value <= 1:
                        all_results.append(f"{value*100:.2f}%")
                    else:
                        all_results.append(f"{value:.2f}")
                    break

    if all_results:
        print(", ".join(all_results))
    else:
        print("N/A (no matching metrics)")

except FileNotFoundError:
    print("N/A (file not found)")
except json.JSONDecodeError:
    print("N/A (invalid JSON)")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
PYTHON_EOF
}

# =============================================================================
# 打印配置
# =============================================================================
echo ""
echo "============================================================"
echo "Batch Model Evaluation"
echo "============================================================"
echo "Task: $TASK"
echo "GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Batch Size (per GPU): $BATCH_SIZE"
echo "Models to evaluate: ${#MODELS[@]}"
echo "============================================================"
echo ""

# 列出要评估的模型
echo "Models:"
for MODEL_NAME in "${!MODELS[@]}"; do
    CKPT_PATH="${MODELS[$MODEL_NAME]}"
    if [[ -d "$CKPT_PATH" ]]; then
        echo "  - $MODEL_NAME: $CKPT_PATH"
    else
        echo "  - $MODEL_NAME: $CKPT_PATH (not found)"
    fi
done
echo ""

# =============================================================================
# 记录所有结果
# =============================================================================
declare -A RESULTS
declare -A ERRORS

# =============================================================================
# 评估每个模型
# =============================================================================
for MODEL_NAME in "${!MODELS[@]}"; do
    CHECKPOINT_PATH="${MODELS[$MODEL_NAME]}"

    echo "------------------------------------------------------------"
    echo "Evaluating: $MODEL_NAME"
    echo "Path: $CHECKPOINT_PATH"
    echo "------------------------------------------------------------"

    if [[ ! -d "$CHECKPOINT_PATH" ]]; then
        echo "Checkpoint不存在，跳过"
        RESULTS[$MODEL_NAME]="N/A (not found)"
        ERRORS[$MODEL_NAME]="Checkpoint not found"
        continue
    fi

    # 验证checkpoint
    if [[ ! -f "$CHECKPOINT_PATH/adapter_config.json" ]]; then
        echo "不是LoRA checkpoint，跳过"
        RESULTS[$MODEL_NAME]="N/A (not LoRA)"
        ERRORS[$MODEL_NAME]="Not a LoRA checkpoint"
        continue
    fi

    if [[ ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" ]] && [[ ! -f "$CHECKPOINT_PATH/adapter_model.bin" ]]; then
        echo "adapter权重不存在，跳过"
        RESULTS[$MODEL_NAME]="N/A (no weights)"
        ERRORS[$MODEL_NAME]="Adapter weights not found"
        continue
    fi

    # 提取base model
    BASE_MODEL=$(python3 -c "
import json
with open('$CHECKPOINT_PATH/adapter_config.json') as f:
    print(json.load(f).get('base_model_name_or_path', ''))
" 2>/dev/null)

    if [[ -z "$BASE_MODEL" ]]; then
        echo "无法提取base model，跳过"
        RESULTS[$MODEL_NAME]="N/A (no base model)"
        ERRORS[$MODEL_NAME]="Could not extract base model"
        continue
    fi

    MODEL_ARGS="pretrained=$BASE_MODEL,peft=$CHECKPOINT_PATH,dtype=bfloat16,trust_remote_code=true"
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${TASK}_${TIMESTAMP}.json"

    echo "开始评估..."

    # 运行评估
    EVAL_SUCCESS=true
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        # 创建临时accelerate配置
        ACCELERATE_CONFIG=$(mktemp)
        cat > "$ACCELERATE_CONFIG" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: $NUM_GPUS
machine_rank: 0
num_machines: 1
gpu_ids: all
mixed_precision: bf16
downcast_bf16: no
use_cpu: false
EOF

        if ! accelerate launch \
            --config_file "$ACCELERATE_CONFIG" \
            -m lm_eval \
            --model hf \
            --model_args "$MODEL_ARGS" \
            --tasks "$TASK" \
            --batch_size "$BATCH_SIZE" \
            --output_path "$OUTPUT_FILE" \
            2>&1; then
            EVAL_SUCCESS=false
        fi

        rm -f "$ACCELERATE_CONFIG"
    else
        if ! python3 -m lm_eval \
            --model hf \
            --model_args "$MODEL_ARGS" \
            --tasks "$TASK" \
            --batch_size "$BATCH_SIZE" \
            --device cuda:0 \
            --output_path "$OUTPUT_FILE" \
            2>&1; then
            EVAL_SUCCESS=false
        fi
    fi

    # 提取结果
    if [[ "$EVAL_SUCCESS" == "true" ]] && [[ -f "$OUTPUT_FILE" ]]; then
        ACC=$(extract_results "$OUTPUT_FILE")
        RESULTS[$MODEL_NAME]="$ACC"
        echo "完成: $ACC"
    else
        RESULTS[$MODEL_NAME]="Error"
        ERRORS[$MODEL_NAME]="Evaluation failed"
        echo "评估失败"
    fi

    # 清理GPU缓存
    cleanup_gpu

    echo ""
done

# =============================================================================
# 生成对比表格
# =============================================================================
echo ""
echo "============================================================"
echo "Results Summary ($TASK)"
echo "============================================================"
printf "%-30s | %-20s\n" "Model" "Accuracy"
echo "-------------------------------|---------------------"

# 按模型名排序输出
for MODEL_NAME in $(printf '%s\n' "${!RESULTS[@]}" | sort); do
    printf "%-30s | %-20s\n" "$MODEL_NAME" "${RESULTS[$MODEL_NAME]}"
done

echo "============================================================"
echo "结束时间: $(date)"
echo ""

# 显示错误（如果有）
if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo "错误信息:"
    for MODEL_NAME in "${!ERRORS[@]}"; do
        echo "  - $MODEL_NAME: ${ERRORS[$MODEL_NAME]}"
    done
    echo ""
fi

echo "详细结果目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"

# =============================================================================
# 保存汇总到单独文件
# =============================================================================
{
    echo "============================================================"
    echo "Batch Evaluation Summary"
    echo "============================================================"
    echo "Task: $TASK"
    echo "GPUs: $NUM_GPUS"
    echo "Time: $(date)"
    echo ""
    printf "%-30s | %-20s\n" "Model" "Accuracy"
    echo "-------------------------------|---------------------"
    for MODEL_NAME in $(printf '%s\n' "${!RESULTS[@]}" | sort); do
        printf "%-30s | %-20s\n" "$MODEL_NAME" "${RESULTS[$MODEL_NAME]}"
    done
    echo "============================================================"

    if [[ ${#ERRORS[@]} -gt 0 ]]; then
        echo ""
        echo "Errors:"
        for MODEL_NAME in "${!ERRORS[@]}"; do
            echo "  - $MODEL_NAME: ${ERRORS[$MODEL_NAME]}"
        done
    fi
} > "$SUMMARY_FILE"

echo "汇总文件: $SUMMARY_FILE"
echo ""
echo "批量评估完成"
