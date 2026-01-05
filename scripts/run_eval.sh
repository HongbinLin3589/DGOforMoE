#!/bin/bash
# =============================================================================
# LM-Evaluation-Harness Evaluation Script for DGO Experiments
# =============================================================================
# 使用方法:
#   bash run_eval.sh <checkpoint_path> [task] [batch_size]
#
# 示例:
#   bash run_eval.sh /path/to/checkpoint-1000 gsm8k_cot 4
#   bash run_eval.sh /path/to/checkpoint-1000 gsm8k
#   bash run_eval.sh /path/to/checkpoint-1000  # 默认: gsm8k_cot
#
# 支持的任务:
#   gsm8k          - GSM8K 直接回答
#   gsm8k_cot      - GSM8K Chain-of-Thought (推荐用于推理模型)
#   gsm8k_cot_zeroshot - GSM8K CoT 零样本
# =============================================================================

set -euo pipefail  # 严格错误处理

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
CHECKPOINT_PATH="${1:-}"
TASK="${2:-gsm8k_cot}"
BATCH_SIZE="${3:-auto}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "请指定checkpoint路径"
    echo ""
    echo "使用方法: bash run_eval.sh <checkpoint_path> [task] [batch_size]"
    echo ""
    echo "示例:"
    echo "  bash run_eval.sh outputs/swift_sft/olmoe_gsm8k/v0-20260104-050951/checkpoint-1000"
    echo "  bash run_eval.sh outputs/swift_grpo/olmoe_gsm8k/v0-20260104-073639/checkpoint-1000 gsm8k_cot 4"
    exit 1
fi

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint路径不存在: $CHECKPOINT_PATH"
    exit 1
fi

# =============================================================================
# 依赖检查
# =============================================================================
echo "检查依赖..."

MISSING_DEPS=()

if ! command -v nvidia-smi &> /dev/null; then
    MISSING_DEPS+=("nvidia-smi")
fi

if ! python3 -c "import lm_eval" 2>/dev/null; then
    MISSING_DEPS+=("lm_eval")
fi

if ! python3 -c "import peft" 2>/dev/null; then
    MISSING_DEPS+=("peft")
fi

if ! python3 -c "import accelerate" 2>/dev/null; then
    MISSING_DEPS+=("accelerate")
fi

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    echo "缺少依赖: ${MISSING_DEPS[*]}"
    echo ""
    echo "请运行以下命令安装:"
    echo "  pip install lm-eval peft accelerate torch transformers"
    exit 1
fi

echo "依赖检查通过"

# =============================================================================
# Checkpoint验证
# =============================================================================
echo ""
echo "验证Checkpoint..."

# 检查是否是LoRA checkpoint
if [[ ! -f "$CHECKPOINT_PATH/adapter_config.json" ]]; then
    echo "不是LoRA checkpoint (缺少 adapter_config.json)"
    exit 1
fi

# 检查adapter权重
if [[ ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" ]] && [[ ! -f "$CHECKPOINT_PATH/adapter_model.bin" ]]; then
    echo "adapter权重文件不存在"

    # 检查是否有DeepSpeed checkpoint
    if ls "$CHECKPOINT_PATH"/global_step* &>/dev/null 2>&1; then
        echo "发现DeepSpeed checkpoint，但adapter权重未合并"
        echo "请先运行: python $CHECKPOINT_PATH/zero_to_fp32.py ..."
    fi
    exit 1
fi

echo "Checkpoint验证通过"

# =============================================================================
# 提取Base Model路径 (带错误处理)
# =============================================================================
BASE_MODEL=$(python3 << PYTHON_EOF
import json
import sys
import os

checkpoint_path = "$CHECKPOINT_PATH"
config_file = os.path.join(checkpoint_path, "adapter_config.json")

if not os.path.exists(config_file):
    print(f"ERROR: {config_file} not found", file=sys.stderr)
    sys.exit(1)

try:
    with open(config_file) as f:
        config = json.load(f)
    base_model = config.get("base_model_name_or_path", "")
    if not base_model:
        print("ERROR: base_model_name_or_path not found", file=sys.stderr)
        sys.exit(1)
    print(base_model)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
)

if [[ $? -ne 0 ]]; then
    echo "无法提取base model路径"
    exit 1
fi

echo "Base Model: $BASE_MODEL"

# =============================================================================
# Batch Size自动调整
# =============================================================================
if [[ "$BATCH_SIZE" == "auto" ]]; then
    # OLMoE-1B-7B 在 80GB GPU 上可以使用更大的 batch_size
    # 根据显存自动调整: 80GB -> 8, 40GB -> 4, 24GB -> 2
    if [[ "$NUM_GPUS" -ge 8 ]]; then
        BATCH_SIZE=8  # 80GB GPU, 可以更激进
    elif [[ "$NUM_GPUS" -ge 4 ]]; then
        BATCH_SIZE=8
    else
        BATCH_SIZE=8
    fi
    echo "自动设置batch size: $BATCH_SIZE (per GPU)"
fi

TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# =============================================================================
# 生成输出文件名
# =============================================================================
CKPT_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")_$(basename "$CHECKPOINT_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/${CKPT_NAME}_${TASK}_${TIMESTAMP}.json"
LOG_FILE="${LOG_DIR}/${CKPT_NAME}_${TASK}_${TIMESTAMP}.log"

# =============================================================================
# 开始记录日志 (和训练脚本一致)
# =============================================================================
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"

# =============================================================================
# 打印配置信息
# =============================================================================
echo ""
echo "============================================================"
echo "LM-Eval 模型评估"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Base Model: $BASE_MODEL"
echo "Task: $TASK"
echo "GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Batch Size (per GPU): $BATCH_SIZE"
echo "Total Batch Size: $TOTAL_BATCH_SIZE"
echo "Output: $OUTPUT_FILE"
echo "============================================================"
echo ""

# =============================================================================
# 构建模型参数
# =============================================================================
MODEL_ARGS="pretrained=$BASE_MODEL,peft=$CHECKPOINT_PATH,dtype=bfloat16,trust_remote_code=true"

# =============================================================================
# 运行评估
# =============================================================================
echo "开始评估..."

if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "使用 $NUM_GPUS GPUs 并行评估 (accelerate)"

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

    accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        -m lm_eval \
        --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --output_path "$OUTPUT_FILE" \
        --log_samples

    EVAL_EXIT_CODE=$?
    rm -f "$ACCELERATE_CONFIG"
else
    echo "使用单GPU评估"
    python3 -m lm_eval \
        --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --device cuda:0 \
        --output_path "$OUTPUT_FILE" \
        --log_samples

    EVAL_EXIT_CODE=$?
fi

# =============================================================================
# 结果处理
# =============================================================================
echo ""
echo "============================================================"

if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    echo "评估完成!"
else
    echo "评估失败 (exit code: $EVAL_EXIT_CODE)"
    echo "请查看日志: $LOG_FILE"
    exit $EVAL_EXIT_CODE
fi

echo "结果保存到: $OUTPUT_FILE"
echo "日志文件: $LOG_FILE"
echo "结束时间: $(date)"
echo "============================================================"

# =============================================================================
# 显示结果摘要
# =============================================================================
if [[ -f "$OUTPUT_FILE" ]]; then
    echo ""
    echo "结果摘要:"
    python3 << PYTHON_EOF
import json
import sys

output_file = "$OUTPUT_FILE"

try:
    with open(output_file) as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results:
        print("  未找到结果")
        sys.exit(0)

    # 提取指标 (按优先级)
    metric_priority = [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match",
        "acc,none",
        "acc_norm,none",
        "acc"
    ]

    for task, metrics in results.items():
        print(f"\n  Task: {task}")

        # 找到最佳指标
        for metric_name in metric_priority:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)):
                    if 0 <= value <= 1:
                        print(f"    {metric_name}: {value*100:.2f}%")
                    else:
                        print(f"    {metric_name}: {value:.4f}")
                    break

        # 显示标准误差 (如果有)
        for key, value in metrics.items():
            if key.endswith("_stderr") and isinstance(value, (int, float)):
                base_metric = key.replace("_stderr", "")
                if base_metric in [m for m in metric_priority if m in metrics]:
                    print(f"    {key}: {value:.4f}")

except Exception as e:
    print(f"  无法解析结果: {e}")
    sys.exit(1)
PYTHON_EOF
fi

echo ""
echo "评估流程完成"
