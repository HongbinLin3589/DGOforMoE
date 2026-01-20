#!/bin/bash
# =============================================================================
# DGO 项目环境配置
# =============================================================================
# 使用方法: source scripts/env.sh
#
# 在其他脚本中使用:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/env.sh"
# =============================================================================

# 获取项目根目录（相对于此脚本）
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    _ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export DGO_ROOT="$(cd "${_ENV_SCRIPT_DIR}/.." && pwd)"
else
    # 如果直接执行而不是 source，使用 PWD
    export DGO_ROOT="$(pwd)"
fi

# =============================================================================
# HuggingFace 配置
# =============================================================================
export HF_HOME="/usr/storage/fwan/huggingface_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ENDPOINT="https://hf-mirror.com"
export USE_HF=1

# =============================================================================
# 模型路径（本地缓存）
# =============================================================================
export MODEL_CACHE="${HF_HUB_CACHE}"

# MoE 模型路径
export OLMOE_MODEL="${MODEL_CACHE}/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654"
export QWEN_MOE_MODEL="${MODEL_CACHE}/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
export DEEPSEEK_MOE_MODEL="${MODEL_CACHE}/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
export MIXTRAL_MODEL="${MODEL_CACHE}/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0"

# 模型名称到路径的映射函数
get_model_path() {
    local model_key="$1"
    case "$model_key" in
        olmoe)    echo "$OLMOE_MODEL" ;;
        qwen)     echo "$QWEN_MOE_MODEL" ;;
        deepseek) echo "$DEEPSEEK_MOE_MODEL" ;;
        mixtral)  echo "$MIXTRAL_MODEL" ;;
        *)        echo "" ;;
    esac
}

# =============================================================================
# 项目目录结构
# =============================================================================
export DGO_SCRIPTS="${DGO_ROOT}/scripts"
export DGO_DATASETS="${DGO_ROOT}/datasets"

# 输出路径配置（可修改为其他磁盘以存储大文件）
# 默认使用 /root/autodl-fs，可通过环境变量覆盖
export DGO_STORAGE="${DGO_STORAGE:-/root/autodl-fs}"
export DGO_OUTPUTS="${DGO_STORAGE}/outputs"
export DGO_LOGS="${DGO_STORAGE}/logs"
export DGO_CACHE="${DGO_STORAGE}/dgo_cache"

# 输出子目录
export SFT_OUTPUT="${DGO_OUTPUTS}/swift_sft"
export GRPO_OUTPUT="${DGO_OUTPUTS}/swift_grpo"
export DGO_OUTPUT="${DGO_OUTPUTS}/swift_dgo"

# 日志子目录
export SFT_LOGS="${DGO_LOGS}/swift_sft"
export GRPO_LOGS="${DGO_LOGS}/swift_grpo"
export DGO_LOGS_DIR="${DGO_LOGS}/swift_dgo"

# =============================================================================
# 数据集路径
# =============================================================================
# HuggingFace 数据集名称
export GSM8K_HF="gsm8k"
export MATH_HF="EleutherAI/hendrycks_math"
export MBPP_HF="google-research-datasets/mbpp"

# 本地数据集路径
export MBPP_LOCAL="${DGO_DATASETS}/mbpp_json/train.json"

# 数据集名称到路径的映射函数
get_dataset_path() {
    local dataset_key="$1"
    case "$dataset_key" in
        gsm8k) echo "$GSM8K_HF" ;;
        math)  echo "$MATH_HF" ;;
        mbpp)  echo "$MBPP_LOCAL" ;;
        *)     echo "" ;;
    esac
}

# =============================================================================
# GPU 配置
# =============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 100))}"

# =============================================================================
# Conda 环境
# =============================================================================
export DGO_CONDA_ENV="DGO"
export CONDA_BASE="/opt/miniforge3"

# 激活 conda 环境的函数
activate_dgo_env() {
    if [[ -f "${CONDA_BASE}/bin/activate" ]]; then
        source "${CONDA_BASE}/bin/activate" "${DGO_CONDA_ENV}"
        echo "✅ 已激活 conda 环境: ${DGO_CONDA_ENV}"
    else
        echo "⚠️  找不到 conda: ${CONDA_BASE}/bin/activate"
    fi
}

# =============================================================================
# 训练超参数默认值
# =============================================================================
export DEFAULT_LORA_RANK=8
export DEFAULT_LORA_ALPHA=64
export DEFAULT_LORA_DROPOUT=0.05
export DEFAULT_LEARNING_RATE="5e-6"
export DEFAULT_WEIGHT_DECAY=0.1
export DEFAULT_WARMUP_RATIO=0.1
export DEFAULT_NUM_EPOCHS=50

# =============================================================================
# MoE 监控默认值
# =============================================================================
export DEFAULT_ROUTER_AUX_LOSS_COEF=0.001
export DEFAULT_MOE_MONITOR_ENABLED=true
export DEFAULT_MOE_LOG_EVERY=10

# =============================================================================
# 辅助函数
# =============================================================================

# 创建必要的目录
ensure_dirs() {
    mkdir -p "${DGO_OUTPUTS}" "${DGO_LOGS}" "${DGO_CACHE}"
    mkdir -p "${SFT_OUTPUT}" "${GRPO_OUTPUT}" "${DGO_OUTPUT}"
    mkdir -p "${SFT_LOGS}" "${GRPO_LOGS}" "${DGO_LOGS_DIR}"
}

# 打印环境配置
print_env() {
    echo "============================================================"
    echo "DGO 环境配置"
    echo "============================================================"
    echo "项目根目录:    ${DGO_ROOT}"
    echo "HF 缓存目录:   ${HF_HOME}"
    echo "HF 镜像源:     ${HF_ENDPOINT}"
    echo ""
    echo "模型路径:"
    echo "  OLMoE:       ${OLMOE_MODEL}"
    echo "  Qwen-MoE:    ${QWEN_MOE_MODEL}"
    echo "  DeepSeek:    ${DEEPSEEK_MOE_MODEL}"
    echo "  Mixtral:     ${MIXTRAL_MODEL}"
    echo ""
    echo "GPU 配置:"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "  NPROC_PER_NODE:       ${NPROC_PER_NODE}"
    echo "============================================================"
}

# 验证模型路径
validate_model() {
    local model_key="$1"
    local model_path=$(get_model_path "$model_key")

    if [[ -z "$model_path" ]]; then
        echo "❌ 未知模型: $model_key"
        echo "可用模型: olmoe, qwen, deepseek, mixtral"
        return 1
    fi

    if [[ ! -d "$model_path" ]]; then
        echo "❌ 模型路径不存在: $model_path"
        return 1
    fi

    echo "✅ 模型验证通过: $model_key"
    return 0
}

# =============================================================================
# 自动初始化
# =============================================================================

# 如果是交互式 shell，打印简短信息
if [[ $- == *i* ]]; then
    echo "DGO 环境已加载 (DGO_ROOT=${DGO_ROOT})"
fi
