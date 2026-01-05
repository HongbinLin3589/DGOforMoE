#!/bin/bash
# =============================================================================
# Quick Test Evaluation - 快速测试评估设置是否正确
# =============================================================================
# 使用方法:
#   bash run_eval_test.sh <checkpoint_path> [task] [limit]
#
# 示例:
#   bash run_eval_test.sh /path/to/checkpoint-1000 gsm8k_cot 10
#   bash run_eval_test.sh /path/to/checkpoint-1000  # 默认10个样本
#
# 说明:
#   只评估少量样本，用于快速验证设置是否正确
#   使用单卡测试，避免占用过多资源
# =============================================================================

set -euo pipefail

# =============================================================================
# 环境配置 (和训练脚本一致)
# =============================================================================
export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="/usr/storage/fwan/huggingface_cache"
export HF_HUB_CACHE="/usr/storage/fwan/huggingface_cache/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

source /opt/miniforge3/bin/activate dgo 2>/dev/null || true

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR="/usr/commondata/public/hf_hub/cc/DGO"
LOG_DIR="${BASE_DIR}/logs/eval"
mkdir -p "$LOG_DIR"

# =============================================================================
# 参数解析
# =============================================================================
CHECKPOINT_PATH="${1:-}"
TASK="${2:-gsm8k_cot}"
LIMIT="${3:-10}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "请指定checkpoint路径"
    echo ""
    echo "使用方法: bash run_eval_test.sh <checkpoint_path> [task] [limit]"
    echo ""
    echo "示例:"
    echo "  bash run_eval_test.sh outputs/swift_sft/olmoe_gsm8k/v0-20260104-050951/checkpoint-1000"
    echo "  bash run_eval_test.sh outputs/swift_grpo/olmoe_gsm8k/v0-20260104-073639/checkpoint-1000 gsm8k_cot 20"
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

if ! python3 -c "import lm_eval" 2>/dev/null; then
    MISSING_DEPS+=("lm_eval")
fi

if ! python3 -c "import peft" 2>/dev/null; then
    MISSING_DEPS+=("peft")
fi

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    echo "缺少依赖: ${MISSING_DEPS[*]}"
    echo "请运行: pip install lm-eval peft"
    exit 1
fi

echo "依赖检查通过"

# =============================================================================
# Checkpoint验证
# =============================================================================
echo ""
echo "验证Checkpoint..."

if [[ ! -f "$CHECKPOINT_PATH/adapter_config.json" ]]; then
    echo "不是LoRA checkpoint (缺少 adapter_config.json)"
    exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" ]] && [[ ! -f "$CHECKPOINT_PATH/adapter_model.bin" ]]; then
    echo "adapter权重文件不存在"
    exit 1
fi

echo "Checkpoint验证通过"

# =============================================================================
# 提取Base Model路径
# =============================================================================
BASE_MODEL=$(python3 << PYTHON_EOF
import json
import sys
import os

checkpoint_path = "$CHECKPOINT_PATH"
config_file = os.path.join(checkpoint_path, "adapter_config.json")

try:
    with open(config_file) as f:
        config = json.load(f)
    base_model = config.get("base_model_name_or_path", "")
    if not base_model:
        print("ERROR: base_model_name_or_path not found", file=sys.stderr)
        sys.exit(1)
    print(base_model)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
)

if [[ $? -ne 0 ]]; then
    echo "无法提取base model路径"
    exit 1
fi

# =============================================================================
# 生成日志文件名和输出文件名
# =============================================================================
CKPT_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")_$(basename "$CHECKPOINT_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${CKPT_NAME}_${TASK}_test_${TIMESTAMP}.log"
# 测试结果放在 eval_results/test 子目录下
OUTPUT_DIR="${BASE_DIR}/eval_results/test"
OUTPUT_FILE="${OUTPUT_DIR}/${CKPT_NAME}_${TASK}_${TIMESTAMP}.json"
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# 开始记录日志 (和训练脚本一致)
# =============================================================================
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"

echo ""
echo "============================================================"
echo "Quick Test Evaluation"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Base Model: $BASE_MODEL"
echo "Task: $TASK"
echo "Limit: $LIMIT samples"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"
echo ""

# =============================================================================
# 构建模型参数
# =============================================================================
MODEL_ARGS="pretrained=$BASE_MODEL,peft=$CHECKPOINT_PATH,dtype=bfloat16,trust_remote_code=true"

# =============================================================================
# 运行测试
# =============================================================================
echo "开始测试评估..."

python3 -m lm_eval \
    --model hf \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --limit "$LIMIT" \
    --batch_size 1 \
    --device cuda:0 \
    --output_path "$OUTPUT_FILE" \
    --log_samples \
    --verbosity INFO

EVAL_EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    echo "测试完成!"
    echo "结果文件: $OUTPUT_FILE"
    echo "日志文件: $LOG_FILE"
    echo "结束时间: $(date)"
    echo ""
    echo "如果没有错误，可以使用 run_eval.sh 进行完整评估:"
    echo "  bash scripts/run_eval.sh $CHECKPOINT_PATH $TASK"
else
    echo "测试失败 (exit code: $EVAL_EXIT_CODE)"
    echo "请查看日志: $LOG_FILE"
fi
echo "============================================================"

exit $EVAL_EXIT_CODE
