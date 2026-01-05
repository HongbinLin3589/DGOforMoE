#!/bin/bash
# =============================================================================
# DGO Full Pipeline (Generation + Training)
# =============================================================================
# 使用方法:
#   bash run_dgo_pipeline.sh [MODEL_NAME] [DATASET_NAME] [MODE] [--freeze_router]
#
# 模式:
#   full  - 运行完整流程: 生成 + 训练 (默认)
#   gen   - 只运行生成阶段
#   train - 只运行训练阶段 (需要已有生成数据)
#
# 示例:
#   bash run_dgo_pipeline.sh olmoe gsm8k                    # 完整流程
#   bash run_dgo_pipeline.sh olmoe gsm8k full --freeze_router  # Group D
#   bash run_dgo_pipeline.sh qwen math gen                  # 只生成
#   bash run_dgo_pipeline.sh deepseek mbpp train            # 只训练
#
# 说明:
#   DGO是两阶段offline RL方法:
#   1. Generation: 用vLLM为每个prompt生成8个response (inference only)
#   2. Training: 用weighted SFT训练模型
# =============================================================================

set -e

# =============================================================================
# 参数解析
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_KEY="${1:-olmoe}"
DATASET_KEY="${2:-gsm8k}"
MODE="${3:-full}"
EXTRA_ARGS=""

# 收集额外参数 (如 --freeze_router)
shift 3 2>/dev/null || true
for arg in "$@"; do
    EXTRA_ARGS="$EXTRA_ARGS $arg"
done

# =============================================================================
# 验证参数
# =============================================================================
if [[ ! "$MODE" =~ ^(full|gen|train)$ ]]; then
    echo "❌ 无效模式: $MODE"
    echo "有效模式: full, gen, train"
    exit 1
fi

# =============================================================================
# 开始
# =============================================================================
echo "============================================================"
echo "         DGO Pipeline"
echo "============================================================"
echo "模型: $MODEL_KEY"
echo "数据集: $DATASET_KEY"
echo "模式: $MODE"
echo "额外参数: $EXTRA_ARGS"
echo "开始时间: $(date)"
echo "============================================================"
echo ""

# =============================================================================
# Phase 1: 数据生成 (Inference Only)
# =============================================================================
if [[ "$MODE" == "full" || "$MODE" == "gen" ]]; then
    echo ""
    echo "================================================================"
    echo "Phase 1: 数据生成 (vLLM Inference)"
    echo "================================================================"
    echo ""

    if bash "$SCRIPT_DIR/run_dgo_gen.sh" "$MODEL_KEY" "$DATASET_KEY"; then
        echo ""
        echo "✅ 生成阶段完成"
    else
        echo ""
        echo "❌ 生成阶段失败"
        exit 1
    fi

    # 如果只做生成，在这里退出
    if [[ "$MODE" == "gen" ]]; then
        echo ""
        echo "============================================================"
        echo "✅ DGO 生成完成"
        echo "结束时间: $(date)"
        echo "============================================================"
        echo ""
        echo "下一步: 运行训练"
        echo "  bash scripts/run_dgo_train.sh $MODEL_KEY $DATASET_KEY"
        exit 0
    fi
fi

# =============================================================================
# Phase 2: 训练 (Offline Weighted SFT)
# =============================================================================
if [[ "$MODE" == "full" || "$MODE" == "train" ]]; then
    echo ""
    echo "================================================================"
    echo "Phase 2: 训练 (Offline Weighted SFT)"
    echo "================================================================"
    echo ""

    # 检查数据是否存在 (train-only模式)
    if [[ "$MODE" == "train" ]]; then
        DGO_DATA_FILE="$BASE_DIR/dgo_cache/dgo_data_${MODEL_KEY}_${DATASET_KEY}.json"
        if [[ ! -f "$DGO_DATA_FILE" ]]; then
            echo "❌ DGO数据文件不存在: $DGO_DATA_FILE"
            echo "请先运行生成: bash $0 $MODEL_KEY $DATASET_KEY gen"
            exit 1
        fi
        echo "✅ 使用已有数据: $DGO_DATA_FILE"
    fi

    if bash "$SCRIPT_DIR/run_dgo_train.sh" "$MODEL_KEY" "$DATASET_KEY" $EXTRA_ARGS; then
        echo ""
        echo "✅ 训练阶段完成"
    else
        echo ""
        echo "❌ 训练阶段失败"
        exit 1
    fi
fi

# =============================================================================
# 总结
# =============================================================================
echo ""
echo "============================================================"
echo "✅ DGO Pipeline 完成!"
echo "============================================================"
echo "模型: $MODEL_KEY"
echo "数据集: $DATASET_KEY"
echo ""
echo "输出位置:"
echo "  数据: $BASE_DIR/dgo_cache/dgo_data_${MODEL_KEY}_${DATASET_KEY}.json"
if [[ "$EXTRA_ARGS" == *"--freeze_router"* ]]; then
    echo "  模型: $BASE_DIR/outputs/swift_dgo/${MODEL_KEY}_${DATASET_KEY}_frozen/"
else
    echo "  模型: $BASE_DIR/outputs/swift_dgo/${MODEL_KEY}_${DATASET_KEY}/"
fi
echo "  日志: $BASE_DIR/logs/"
echo ""
echo "结束时间: $(date)"
echo "============================================================"
echo ""
echo "下一步:"
echo ""
echo "1. 评估DGO模型:"
echo "   lm-eval run --model hf \\"
echo "     --model_args pretrained=$BASE_DIR/outputs/swift_dgo/${MODEL_KEY}_${DATASET_KEY} \\"
echo "     --tasks gsm8k --batch_size auto"
echo ""
echo "2. 对比GRPO基线:"
echo "   bash scripts/run_grpo_swift.sh $MODEL_KEY $DATASET_KEY"
echo ""
echo "3. 合并LoRA (可选):"
echo "   swift export --ckpt_dir outputs/swift_dgo/${MODEL_KEY}_${DATASET_KEY} --merge_lora true"
echo ""
echo "4. Group D实验 (冻结router):"
echo "   bash scripts/run_dgo_pipeline.sh $MODEL_KEY $DATASET_KEY train --freeze_router"
echo "============================================================"
