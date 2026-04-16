#!/bin/bash
# =============================================================================
# 批量评测 qwen bigmath 所有 checkpoints (SFT / GRPO / DGO)
# 2-GPU 并行：GPU1 + GPU2 同时跑不同 checkpoint（TP=1, DP=2）
# 直接用 LoRA adapter，不 merge
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
activate_dgo_env

export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="https://hf-mirror.com"
export LD_PRELOAD="/opt/miniforge3/envs/DGO/lib/libstdc++.so.6:${LD_PRELOAD}"

GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"

BASE_MODEL="${QWEN_MOE_MODEL}"
EVAL_DIR="${DGO_STORAGE}/eval/qwen_bigmath_all"
LOG_DIR="${DGO_LOGS}/eval"
PLOT_DIR="${DGO_ROOT}/Plot"

mkdir -p "$EVAL_DIR" "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 列表
# ─────────────────────────────────────────────────────────────────────────────
SFT_DIR="/usr/storage/fwan/DGO/outputs/swift_sft/qwen_bigmath/v5-20260328-173809"
GRPO_DIR="/usr/commondata/public/hf_hub/cc/DGOforMoE/outputs/swift_grpo/qwen_bigmath/v2-20260328-224328"
DGO_DIR="/usr/commondata/public/hf_hub/cc/DGOforMoE/outputs/swift_dgo/qwen_bigmath/v3-20260405-155724"

# 所有 (train_type, checkpoint_dir, step) 三元组
ALL_CKPTS=(
    "sft:${SFT_DIR}/checkpoint-100:100"
    "sft:${SFT_DIR}/checkpoint-200:200"
    "sft:${SFT_DIR}/checkpoint-300:300"
    "sft:${SFT_DIR}/checkpoint-400:400"
    "sft:${SFT_DIR}/checkpoint-500:500"
    "sft:${SFT_DIR}/checkpoint-600:600"
    "sft:${SFT_DIR}/checkpoint-700:700"
    "sft:${SFT_DIR}/checkpoint-800:800"
    "sft:${SFT_DIR}/checkpoint-900:900"
    "sft:${SFT_DIR}/checkpoint-1000:1000"
    "grpo:${GRPO_DIR}/checkpoint-100:100"
    "grpo:${GRPO_DIR}/checkpoint-200:200"
    "grpo:${GRPO_DIR}/checkpoint-300:300"
    "grpo:${GRPO_DIR}/checkpoint-400:400"
    "grpo:${GRPO_DIR}/checkpoint-500:500"
    "grpo:${GRPO_DIR}/checkpoint-600:600"
    "grpo:${GRPO_DIR}/checkpoint-700:700"
    "grpo:${GRPO_DIR}/checkpoint-800:800"
    "grpo:${GRPO_DIR}/checkpoint-900:900"
    "grpo:${GRPO_DIR}/checkpoint-1000:1000"
    "grpo:${GRPO_DIR}/checkpoint-1100:1100"
    "grpo:${GRPO_DIR}/checkpoint-1200:1200"
    "grpo:${GRPO_DIR}/checkpoint-1248:1248"
    "dgo:${DGO_DIR}/checkpoint-100:100"
    "dgo:${DGO_DIR}/checkpoint-200:200"
    "dgo:${DGO_DIR}/checkpoint-300:300"
    "dgo:${DGO_DIR}/checkpoint-400:400"
    "dgo:${DGO_DIR}/checkpoint-500:500"
    "dgo:${DGO_DIR}/checkpoint-600:600"
    "dgo:${DGO_DIR}/checkpoint-700:700"
    "dgo:${DGO_DIR}/checkpoint-800:800"
    "dgo:${DGO_DIR}/checkpoint-900:900"
    "dgo:${DGO_DIR}/checkpoint-1000:1000"
    "dgo:${DGO_DIR}/checkpoint-1100:1100"
    "dgo:${DGO_DIR}/checkpoint-1200:1200"
    "dgo:${DGO_DIR}/checkpoint-1252:1252"
)

TOTAL=${#ALL_CKPTS[@]}

LOG_FILE="${LOG_DIR}/eval_qwen_bigmath_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " qwen bigmath 批量评测（2-GPU 并行）"
echo " GPU1=$GPU1  GPU2=$GPU2  TP=1"
echo " 总 checkpoints: $TOTAL"
echo " 日志: $LOG_FILE"
echo "============================================================"
echo "开始时间: $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# 单个 checkpoint 推理函数（在指定 GPU 上运行）
# ─────────────────────────────────────────────────────────────────────────────
run_one_on_gpu() {
    local gpu_id="$1"
    local train_type="$2"
    local ckpt_dir="$3"
    local step="$4"

    # SFT / GRPO / DGO 统一使用 no-CoT prompt（仅 <answer>\boxed{}），研究约束：直接答题不允许思考
    local result_json="${ckpt_dir}/eval_result.json"
    local sys_prompt_arg=""

    local raw_json="${EVAL_DIR}/raw_${train_type}_step${step}.json"
    local ckpt_log="${LOG_DIR}/eval_${train_type}_step${step}.log"

    if [[ -f "$result_json" ]]; then
        echo "  [GPU$gpu_id] ⏭  已有结果跳过: ${train_type} step=${step}"
        return 0
    fi

    if [[ ! -d "$ckpt_dir" ]]; then
        echo "  [GPU$gpu_id] ❌ checkpoint 不存在: $ckpt_dir"
        return 0
    fi

    echo "  [GPU$gpu_id] 🚀 ${train_type^^} step=${step} → $raw_json"

    CUDA_VISIBLE_DEVICES=$gpu_id python "${DGO_ROOT}/vllm_inference.py" \
        --model_name "$BASE_MODEL" \
        --lora_path "$ckpt_dir" \
        --max_lora_rank 64 \
        --dataset bigmath \
        --dataset_split test \
        --n 1 \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 1024 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --max_num_seqs 256 \
        --max_num_batched_tokens 32768 \
        --swap_space 8 \
        --output_file "$raw_json" \
        $sys_prompt_arg \
        --stop '</answer>' \
        --stop $'\nQ:' \
        --stop $'\n\nQ:' \
        > "$ckpt_log" 2>&1

    # 计算准确率
    python3 - <<PYEOF >> "$ckpt_log" 2>&1
import json, re, sys

try:
    from math_verify import parse, verify
    USE_MV = True
except ImportError:
    USE_MV = False

def extract_boxed(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    s = m.group(1) if m else text
    idx = s.rfind(r'\\boxed{')
    if idx == -1:
        return None
    d = 0
    for i in range(s.index('{', idx), len(s)):
        if s[i] == '{': d += 1
        elif s[i] == '}':
            d -= 1
            if d == 0:
                return s[s.index('{', idx)+1:i].strip()
    return None

def check(pred, gt):
    if pred is None or gt is None: return False
    if USE_MV:
        try: return verify(parse(pred), parse(gt))
        except: pass
    p = re.sub(r'\s+', ' ', pred.strip().rstrip('.'))
    g = re.sub(r'\s+', ' ', gt.strip().rstrip('.'))
    return p == g

with open('${raw_json}') as f:
    data = json.load(f)

total, correct, no_answer = len(data), 0, 0
details = []
for item in data:
    completion = item['generated_texts'][0]
    gt_raw = item['ground_truth_answer']
    pred = extract_boxed(completion)
    gt = extract_boxed(gt_raw)
    if gt is None: gt = gt_raw.strip()
    ok = check(pred, gt)
    if ok: correct += 1
    if pred is None: no_answer += 1
    details.append({'prompt': item['prompt'][:80]+'...', 'ground_truth': gt_raw[:60],
                    'predicted': pred, 'correct': ok})

accuracy = correct / total * 100 if total > 0 else 0
result = {
    'train_type': '${train_type}',
    'step': ${step},
    'model': 'qwen',
    'dataset': 'bigmath',
    'split': 'test',
    'total': total,
    'correct': correct,
    'accuracy': accuracy,
    'no_answer': no_answer,
    'details': details,
}
with open('${result_json}', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f'✅ ${train_type} step=${step}: accuracy={accuracy:.2f}% ({correct}/{total}) no_ans={no_answer}')
PYEOF

    local acc=$(python3 -c "import json; d=json.load(open('${result_json}')); print(f\"{d['accuracy']:.2f}%\")" 2>/dev/null || echo "?")
    echo "  [GPU$gpu_id] ✅ ${train_type^^} step=${step}  acc=$acc"
}

# ─────────────────────────────────────────────────────────────────────────────
# 2-GPU 并行主循环：每轮同时跑 2 个 checkpoint
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo " 开始 2-GPU 并行推理..."
echo ""

idx=0
while [ $idx -lt $TOTAL ]; do
    entry1="${ALL_CKPTS[$idx]}"
    entry2="${ALL_CKPTS[$((idx+1))]:-}"

    IFS=':' read -r tt1 ckpt1 step1 <<< "$entry1"

    # GPU1 跑
    if [[ -f "${ckpt1}/eval_result.json" ]]; then
        echo "  [GPU$GPU1] ⏭  已有结果跳过: ${tt1} step=${step1}"
        pid1=""
    else
        run_one_on_gpu "$GPU1" "$tt1" "$ckpt1" "$step1" &
        pid1=$!
    fi

    # GPU2 跑（如果还有 checkpoint）
    pid2=""
    if [[ -n "$entry2" ]]; then
        IFS=':' read -r tt2 ckpt2 step2 <<< "$entry2"
        if [[ -f "${ckpt2}/eval_result.json" ]]; then
            echo "  [GPU$GPU2] ⏭  已有结果跳过: ${tt2} step=${step2}"
        else
            run_one_on_gpu "$GPU2" "$tt2" "$ckpt2" "$step2" &
            pid2=$!
        fi
    fi

    # 等待本轮完成
    [[ -n "$pid1" ]] && wait $pid1
    [[ -n "$pid2" ]] && wait $pid2

    idx=$((idx + 2))
    echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# 汇总结果
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " 汇总结果"
echo "============================================================"
for entry in "${ALL_CKPTS[@]}"; do
    IFS=':' read -r tt ckpt step <<< "$entry"
    # 统一 no-CoT 流程后，SFT / GRPO / DGO 所有 train_type 共用 eval_result.json
    rf="${ckpt}/eval_result.json"
    label="${tt^^}"
    if [[ -f "$rf" ]]; then
        acc=$(python3 -c "import json; d=json.load(open('$rf')); print(f\"{d['accuracy']:.2f}%\")" 2>/dev/null || echo "?")
        printf "  %-12s step=%4s  %s\n" "$label" "$step" "$acc"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " 绘图..."
echo "============================================================"
python "${DGO_ROOT}/Plot/plot_accuracy.py" \
    --scan_dirs "$SFT_DIR" "$GRPO_DIR" "$DGO_DIR" \
    --save_dir "$PLOT_DIR"

echo ""
echo "============================================================"
echo " ✅ 全部完成！"
echo " 结果目录: $EVAL_DIR"
echo " 图片目录: $PLOT_DIR"
echo " 结束时间: $(date)"
echo "============================================================"
