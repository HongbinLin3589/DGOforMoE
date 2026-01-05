#!/bin/bash
# =============================================================================
# validate_checkpoint.sh - Checkpoint验证工具
# =============================================================================
# 用法: bash validate_checkpoint.sh <checkpoint_path>
#
# 功能:
#   1. 检查checkpoint是否是LoRA格式
#   2. 验证adapter_config.json
#   3. 检查adapter权重文件
#   4. 检测DeepSpeed状态
#   5. 验证base model路径
# =============================================================================

set -euo pipefail

CHECKPOINT_PATH="${1:-}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "用法: bash validate_checkpoint.sh <checkpoint_path>"
    exit 1
fi

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint路径不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "验证Checkpoint: $CHECKPOINT_PATH"
echo ""

# =============================================================================
# 检查是否是LoRA checkpoint
# =============================================================================
if [[ ! -f "$CHECKPOINT_PATH/adapter_config.json" ]]; then
    echo "不是LoRA checkpoint (缺少 adapter_config.json)"
    exit 1
fi

echo "检测到LoRA checkpoint"

# =============================================================================
# 验证adapter_config.json并提取信息
# =============================================================================
VALIDATION_RESULT=$(python3 << PYTHON_EOF
import json
import sys
import os

checkpoint_path = "$CHECKPOINT_PATH"
config_file = os.path.join(checkpoint_path, "adapter_config.json")

try:
    with open(config_file) as f:
        config = json.load(f)

    # 提取关键信息
    base_model = config.get("base_model_name_or_path", "")
    peft_type = config.get("peft_type", "")
    lora_r = config.get("r", "N/A")
    lora_alpha = config.get("lora_alpha", "N/A")
    target_modules = config.get("target_modules", [])

    if not base_model:
        print("ERROR: base_model_name_or_path not found", file=sys.stderr)
        sys.exit(1)

    # 检查base model路径
    base_model_exists = os.path.exists(base_model)

    # 输出信息
    print(f"BASE_MODEL={base_model}")
    print(f"PEFT_TYPE={peft_type}")
    print(f"LORA_R={lora_r}")
    print(f"LORA_ALPHA={lora_alpha}")
    print(f"TARGET_MODULES={','.join(target_modules)}")
    print(f"BASE_MODEL_EXISTS={base_model_exists}")

except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON in adapter_config.json: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
)

if [[ $? -ne 0 ]]; then
    echo "配置文件验证失败"
    echo "$VALIDATION_RESULT"
    exit 1
fi

# 解析验证结果
eval "$VALIDATION_RESULT"

echo ""
echo "Adapter配置:"
echo "  Base Model: $BASE_MODEL"
echo "  PEFT Type: $PEFT_TYPE"
echo "  LoRA Rank: $LORA_R"
echo "  LoRA Alpha: $LORA_ALPHA"
echo "  Target Modules: $TARGET_MODULES"

# 检查base model是否存在
if [[ "$BASE_MODEL_EXISTS" == "True" ]]; then
    echo "  Base Model路径存在"
else
    echo "  Base Model路径不存在: $BASE_MODEL"
    echo "  (可能是HuggingFace model ID，将在评估时下载)"
fi

# =============================================================================
# 检查adapter权重
# =============================================================================
echo ""
echo "Adapter权重:"
ADAPTER_AVAILABLE=false

if [[ -f "$CHECKPOINT_PATH/adapter_model.safetensors" ]]; then
    ADAPTER_SIZE=$(ls -lh "$CHECKPOINT_PATH/adapter_model.safetensors" | awk '{print $5}')
    echo "  adapter_model.safetensors ($ADAPTER_SIZE)"
    ADAPTER_AVAILABLE=true
elif [[ -f "$CHECKPOINT_PATH/adapter_model.bin" ]]; then
    ADAPTER_SIZE=$(ls -lh "$CHECKPOINT_PATH/adapter_model.bin" | awk '{print $5}')
    echo "  adapter_model.bin ($ADAPTER_SIZE)"
    ADAPTER_AVAILABLE=true
else
    echo "  adapter权重文件不存在"
fi

# =============================================================================
# 检查DeepSpeed状态
# =============================================================================
echo ""
echo "DeepSpeed状态:"
DEEPSPEED_DIRS=$(find "$CHECKPOINT_PATH" -maxdepth 1 -type d -name "global_step*" 2>/dev/null || true)

if [[ -n "$DEEPSPEED_DIRS" ]]; then
    DEEPSPEED_SIZE=$(du -sh "$CHECKPOINT_PATH"/global_step* 2>/dev/null | awk '{print $1}' | head -1)
    echo "  检测到DeepSpeed checkpoint ($DEEPSPEED_SIZE)"
    echo "  DeepSpeed优化器状态在评估时将被忽略"

    if [[ "$ADAPTER_AVAILABLE" != "true" ]]; then
        echo ""
        echo "  错误: adapter权重未合并"
        echo "  需要先运行: python $CHECKPOINT_PATH/zero_to_fp32.py ..."
        exit 1
    fi
else
    echo "  未检测到DeepSpeed checkpoint (正常)"
fi

# =============================================================================
# 验证结果
# =============================================================================
echo ""
if [[ "$ADAPTER_AVAILABLE" == "true" ]]; then
    echo "Checkpoint验证通过"
    echo ""
    echo "可以使用以下命令评估:"
    echo "  bash scripts/run_eval.sh $CHECKPOINT_PATH gsm8k_cot"
    exit 0
else
    echo "Checkpoint验证失败: 缺少adapter权重"
    exit 1
fi
