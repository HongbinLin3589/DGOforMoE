#!/bin/bash
# =============================================================================
# 模型评测脚本 — 在测试集上生成回答并计算准确率
# =============================================================================
# 使用方法:
#   bash run_test.sh MODEL_NAME DATASET_NAME [CHECKPOINT]
#
# 模型选项: olmoe, olmoe_instruct, qwen, qwen3, qwen3_instruct, deepseek, mixtral
# 数据集选项: gsm8k, math, mbpp, bigmath
# CHECKPOINT: 可选，LoRA adapter 路径（SFT/DGO/GRPO 均可）
#
# 示例:
#   # --- 基础模型 / Instruct 模型 ---
#   bash run_test.sh olmoe bigmath                                    # OLMoE base
#   bash run_test.sh olmoe_instruct bigmath                           # OLMoE Instruct
#   bash run_test.sh qwen3 bigmath                                    # Qwen3-30B base
#
#   # --- SFT checkpoint ---
#   bash run_test.sh olmoe bigmath outputs/swift_sft/.../checkpoint-1000
#
#   # --- GRPO checkpoint (LoRA adapter，自动合并后推理) ---
#   bash run_test.sh olmoe bigmath outputs/swift_grpo/.../checkpoint-500
#
#   # --- DGO checkpoint ---
#   bash run_test.sh olmoe bigmath outputs/swift_dgo/.../checkpoint-500
#
#   # --- 大模型需要多卡 TP ---
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_test.sh qwen3 bigmath
#   CUDA_VISIBLE_DEVICES=0,1 bash run_test.sh deepseek bigmath
#
# 说明:
#   - CHECKPOINT 支持任意 LoRA adapter（SFT/DGO/GRPO 训练产出）
#   - 脚本自动合并 LoRA 到 base model（swift export --merge_lora）
#   - 合并结果缓存在 ${CHECKPOINT}-merged/，重复运行自动跳过
#   - 文件名自动标记来源: olmoe_sft1000 / olmoe_grpo500 / olmoe_dgo500
#
# 流程:
#   1. LoRA 合并: swift export --merge_lora（如有 checkpoint）
#   2. vLLM 推理: 对测试集每个问题生成 1 个回答 (greedy decoding)
#   3. 准确率计算: math_verify 符号等价比较 + 字符串匹配 fallback
#   4. 输出: 准确率 + 详细结果 JSON → /wutailin/DGO/eval/
# =============================================================================

set -e

# =============================================================================
# 加载环境配置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# 覆盖GPU配置
if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# 激活 conda 环境
activate_dgo_env

export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0

ensure_dirs

# =============================================================================
# 配置
# =============================================================================
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

declare -A MAX_LENGTH
MAX_LENGTH["gsm8k"]=1024
MAX_LENGTH["math"]=1024
MAX_LENGTH["mbpp"]=1024
MAX_LENGTH["bigmath"]=1024

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

# 评测参数: greedy decoding, 每题 1 个回答
NUM_GENERATIONS=1
TEMPERATURE=0.0
TOP_P=1.0
VLLM_MEM=${VLLM_MEM:-0.85}
SWAP_SPACE=8

# =============================================================================
# 参数解析
# =============================================================================
MODEL_KEY="${1:-olmoe}"
DATASET_KEY="${2:-bigmath}"
CHECKPOINT="${3:-}"

MODEL_PATH=$(get_model_path "$MODEL_KEY")
if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ 未知模型: $MODEL_KEY"
    echo "可用模型: olmoe, olmoe_instruct, qwen, qwen3, qwen3_instruct, deepseek, mixtral"
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

case "$DATASET_KEY" in
    gsm8k|math|mbpp|bigmath) ;;
    *)
        echo "❌ 未知数据集: $DATASET_KEY"
        exit 1
        ;;
esac

MAX_LEN="${MAX_LENGTH[$DATASET_KEY]}"
MAX_SEQS="${MAX_NUM_SEQS[$MODEL_KEY]:-256}"
MAX_TOKENS_BATCH="${MAX_BATCHED_TOKENS[$MODEL_KEY]:-16384}"

# 输出路径
EVAL_DIR="${DGO_STORAGE}/eval"
mkdir -p "$EVAL_DIR"

# 确定模型标签（用于文件名）
if [[ -n "$CHECKPOINT" ]]; then
    # 从 checkpoint 路径提取有意义的标签
    # 例如: .../swift_sft/olmoe_bigmath/v1-.../checkpoint-1000 → sft_ckpt1000
    CKPT_STEP=$(basename "$CHECKPOINT" | grep -oP '\d+$' || echo "unknown")
    if [[ "$CHECKPOINT" == *swift_sft* ]]; then
        MODEL_TAG="${MODEL_KEY}_sft${CKPT_STEP}"
    elif [[ "$CHECKPOINT" == *swift_dgo* ]]; then
        MODEL_TAG="${MODEL_KEY}_dgo${CKPT_STEP}"
    elif [[ "$CHECKPOINT" == *swift_grpo* ]]; then
        MODEL_TAG="${MODEL_KEY}_grpo${CKPT_STEP}"
    else
        MODEL_TAG="${MODEL_KEY}_ckpt${CKPT_STEP}"
    fi
else
    MODEL_TAG="${MODEL_KEY}_base"
fi

OUTPUT_FILE="${EVAL_DIR}/eval_${MODEL_TAG}_${DATASET_KEY}.json"
LOG_FILE="${DGO_LOGS}/eval/${MODEL_TAG}_${DATASET_KEY}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

# =============================================================================
# 开始记录日志
# =============================================================================
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Model Evaluation"
echo "============================================================"
echo "模型: $MODEL_KEY ($MODEL_TAG)"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET_KEY (test split)"
echo "Checkpoint: ${CHECKPOINT:-(base model)}"
echo "输出文件: $OUTPUT_FILE"
echo ""
echo "推理配置:"
echo "  num_generations: $NUM_GENERATIONS (greedy)"
echo "  temperature: $TEMPERATURE"
echo "  max_tokens: $MAX_LEN"
echo "============================================================"
echo "开始时间: $(date)"
echo ""

# =============================================================================
# Step 1: 处理 checkpoint（合并 LoRA）
# =============================================================================
INFERENCE_MODEL="$MODEL_PATH"
if [[ -n "$CHECKPOINT" ]]; then
    if [[ ! -d "$CHECKPOINT" ]]; then
        echo "❌ Checkpoint 路径不存在: $CHECKPOINT"
        exit 1
    fi

    MERGED_DIR="${CHECKPOINT}-merged"
    if [[ -d "$MERGED_DIR" && -f "$MERGED_DIR/config.json" ]]; then
        echo "📦 发现已合并的模型: $MERGED_DIR (跳过合并)"
    else
        echo "📦 合并 LoRA adapter 到 base model..."
        swift export \
            --ckpt_dir "$CHECKPOINT" \
            --merge_lora true \
            --output_dir "$MERGED_DIR"
        if [[ ! -d "$MERGED_DIR" ]]; then
            echo "❌ 合并失败"
            exit 1
        fi
        echo "✅ 合并完成"
    fi
    INFERENCE_MODEL="$MERGED_DIR"
fi

# =============================================================================
# Step 2: vLLM 推理（test split, n=1, greedy）
# =============================================================================
# 自动选择 TP 大小：小模型 TP=1，大模型按 GPU 数
case "$MODEL_KEY" in
    olmoe|olmoe_instruct) VLLM_TP=1 ;;
    *)                    VLLM_TP=${TP_SIZE:-$NUM_GPUS} ;;
esac

echo "🚀 开始推理... (TP=$VLLM_TP)"

python "${DGO_ROOT}/vllm_inference.py" \
    --model_name "$INFERENCE_MODEL" \
    --dataset "$DATASET_KEY" \
    --dataset_split test \
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

echo "✅ 推理完成"
echo ""

# =============================================================================
# Step 3: 计算准确率
# =============================================================================
echo "📊 计算准确率..."

python -c "
import json, re, sys

# 使用 math_verify 进行符号等价比较（如 \\sqrt{48} == 4\\sqrt{3}）
try:
    from math_verify import parse, verify
    USE_MATH_VERIFY = True
    print('>>> 使用 math_verify 进行符号等价比较')
except ImportError:
    USE_MATH_VERIFY = False
    print('>>> math_verify 未安装，使用字符串精确匹配')

def extract_boxed(text):
    \"\"\"从文本中提取 \\boxed{...} 的内容，支持嵌套大括号。\"\"\"
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    search_text = answer_match.group(1) if answer_match else text

    idx = search_text.rfind('\\\\boxed{')
    if idx == -1:
        return None

    brace_start = search_text.index('{', idx)
    depth = 0
    for i in range(brace_start, len(search_text)):
        if search_text[i] == '{':
            depth += 1
        elif search_text[i] == '}':
            depth -= 1
            if depth == 0:
                return search_text[brace_start+1:i].strip()
    return None

def check_correct(pred_str, gt_str):
    \"\"\"比较预测答案和 ground truth，优先用 math_verify 符号比较。\"\"\"
    if pred_str is None or gt_str is None:
        return False
    if USE_MATH_VERIFY:
        try:
            return verify(parse(pred_str), parse(gt_str))
        except Exception:
            pass  # math_verify 超时或解析失败，fallback 到字符串比较
    # 字符串精确匹配（归一化后）
    p = re.sub(r'\s+', ' ', pred_str.strip().rstrip('.'))
    g = re.sub(r'\s+', ' ', gt_str.strip().rstrip('.'))
    return p == g

# 加载结果
with open('$OUTPUT_FILE') as f:
    data = json.load(f)

total = len(data)
correct = 0
no_answer = 0
timeout_count = 0
details = []

for item in data:
    completion = item['generated_texts'][0]
    gt_raw = item['ground_truth_answer']

    pred = extract_boxed(completion)
    gt = extract_boxed(gt_raw)
    if gt is None:
        gt = gt_raw.strip()

    is_correct = check_correct(pred, gt)
    if is_correct:
        correct += 1
    if pred is None:
        no_answer += 1

    details.append({
        'prompt': item['prompt'][:100] + '...',
        'ground_truth': gt_raw[:80],
        'predicted': pred,
        'correct': is_correct,
    })

accuracy = correct / total * 100 if total > 0 else 0

print(f'')
print(f'============================================================')
print(f'  评测结果: {\"$MODEL_TAG\"} on {\"$DATASET_KEY\"} (test)')
print(f'============================================================')
print(f'  总题数:     {total}')
print(f'  正确:       {correct}')
print(f'  准确率:     {accuracy:.2f}%')
print(f'  无答案:     {no_answer} ({no_answer/total*100:.1f}%)')
print(f'============================================================')

# 保存详细结果
result_file = '${OUTPUT_FILE}'.replace('.json', '_result.json')
result = {
    'model': '$MODEL_TAG',
    'dataset': '$DATASET_KEY',
    'split': 'test',
    'total': total,
    'correct': correct,
    'accuracy': accuracy,
    'no_answer': no_answer,
    'details': details,
}
with open(result_file, 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f'  详细结果:   {result_file}')
print(f'============================================================')
"

echo ""
echo "结束时间: $(date)"
