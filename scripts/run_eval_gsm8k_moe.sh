#!/bin/bash
# =============================================================================
# GSM8K 评测：no-CoT accuracy (base model + 所有 checkpoints)
# 统一 no-CoT prompt；结果写入 eval_result_gsm8k_noCoT.json（不覆盖旧 CoT 结果）
# 多 GPU 并行运行（每轮同时跑 N 个 checkpoint，每个 checkpoint 占一张卡）
# 用法: GPUS="0 1 3 4" bash scripts/run_eval_gsm8k_moe.sh
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

# 支持多卡：GPUS="0 1 3 4" 或单卡 GPUS="2"
IFS=' ' read -r -a GPU_LIST <<< "${GPUS:-0 1 3 4}"
N_GPU=${#GPU_LIST[@]}

BASE_MODEL="${QWEN_MOE_MODEL}"
EVAL_DIR="${DGO_STORAGE}/eval/qwen_gsm8k_all"
LOG_DIR="${DGO_LOGS}/eval"
PLOT_DIR="${DGO_ROOT}/Plot"

mkdir -p "$EVAL_DIR" "$LOG_DIR"

SFT_DIR="/usr/storage/fwan/DGO/outputs/swift_sft/qwen_bigmath/v5-20260328-173809"
GRPO_DIR="/usr/commondata/public/hf_hub/cc/DGOforMoE/outputs/swift_grpo/qwen_bigmath/v2-20260328-224328"
DGO_DIR="/usr/commondata/public/hf_hub/cc/DGOforMoE/outputs/swift_dgo/qwen_bigmath/v3-20260405-155724"

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

# 按 train_type 过滤（默认跑全部；例：TRAIN_TYPES="sft" 只跑 SFT）
TRAIN_TYPES="${TRAIN_TYPES:-sft grpo dgo}"
_FILTERED=()
for _entry in "${ALL_CKPTS[@]}"; do
    IFS=':' read -r _tt _ _ <<< "$_entry"
    for _t in $TRAIN_TYPES; do
        if [[ "$_tt" == "$_t" ]]; then
            _FILTERED+=("$_entry")
            break
        fi
    done
done
ALL_CKPTS=("${_FILTERED[@]}")
TOTAL=${#ALL_CKPTS[@]}
echo "过滤后 train_types=[$TRAIN_TYPES]: $TOTAL 个 checkpoint"

LOG_FILE="${LOG_DIR}/eval_qwen_gsm8k_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " qwen GSM8K Accuracy 评测（${N_GPU}-GPU 并行，含 base model）"
echo " GPUs=${GPU_LIST[*]}"
echo " 总 checkpoints: $TOTAL（+ base model）"
echo " 日志: $LOG_FILE"
echo "============================================================"
echo "开始时间: $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# 单个 checkpoint accuracy 评测（vLLM，单卡）
# ─────────────────────────────────────────────────────────────────────────────
run_acc() {
    local gpu_id="$1"
    local train_type="$2"
    local ckpt_dir="$3"
    local step="$4"

    local result_json="${ckpt_dir}/eval_result_gsm8k_noCoT.json"
    local raw_json="${EVAL_DIR}/raw_${train_type}_step${step}_noCoT.json"
    local ckpt_log="${LOG_DIR}/eval_gsm8k_${train_type}_step${step}_noCoT.log"

    # SFT / GRPO / DGO 统一使用 no-CoT prompt（仅 <answer>\boxed{}），研究约束：直接答题不允许思考
    local sys_prompt_arg=""

    if [[ -f "$result_json" ]]; then
        echo "  [GPU$gpu_id] ⏭  跳过(已有): ${train_type} step=${step}"
        return 0
    fi
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "  [GPU$gpu_id] ❌ 不存在: $ckpt_dir"
        return 0
    fi

    echo "  [GPU$gpu_id] 🚀 ${train_type^^} step=${step}"

    CUDA_VISIBLE_DEVICES=$gpu_id python "${DGO_ROOT}/vllm_inference.py" \
        --model_name "$BASE_MODEL" \
        --lora_path "$ckpt_dir" \
        --max_lora_rank 64 \
        --dataset gsm8k \
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

    python3 - <<PYEOF >> "$ckpt_log" 2>&1
import json, re

def extract_boxed(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    s = m.group(1) if m else text
    idx = s.rfind(r'\\boxed{')
    if idx == -1: return None
    d = 0
    for i in range(s.index('{', idx), len(s)):
        if s[i] == '{': d += 1
        elif s[i] == '}':
            d -= 1
            if d == 0: return s[s.index('{', idx)+1:i].strip()
    return None

def norm(s):
    if s is None: return None
    s = s.replace(',','').strip().rstrip('.')
    try:
        f = float(s); return str(int(f)) if f==int(f) else str(f)
    except: return s

def extract_gt(text):
    if text is None: return None
    m = re.search(r'####\s*([\d,\.]+)', text)
    return m.group(1).replace(',','').strip() if m else text.strip()

with open('${raw_json}') as f:
    data = json.load(f)

total, correct, no_answer = len(data), 0, 0
details = []
for item in data:
    response = item['generated_texts'][0]
    pred = norm(extract_boxed(response))
    gt   = norm(extract_gt(item['ground_truth_answer']))
    ok   = pred is not None and gt is not None and pred == gt
    if ok: correct += 1
    if pred is None: no_answer += 1
    # 提取问题文本（去掉 system prompt，只保留 Q: 之后的部分）
    prompt = item['prompt']
    q_start = prompt.rfind('\nQ:')
    question = prompt[q_start+3:].strip() if q_start != -1 else prompt[-300:].strip()
    details.append({
        'question':       question,
        'response':       response,
        'predicted':      pred,
        'ground_truth':   item['ground_truth_answer'],
        'gt_normalized':  gt,
        'correct':        ok,
    })

accuracy = correct / total * 100 if total > 0 else 0
result = {
    'train_type': '${train_type}', 'step': ${step}, 'model': 'qwen',
    'dataset': 'gsm8k', 'split': 'test', 'total': total,
    'correct': correct, 'accuracy': accuracy, 'no_answer': no_answer,
    'details': details,
}
with open('${result_json}','w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f'✅ ${train_type} step=${step}: {accuracy:.2f}% ({correct}/{total}) no_ans={no_answer}')
PYEOF

    local acc
    acc=$(python3 -c "import json; d=json.load(open('${result_json}')); print(f\"{d['accuracy']:.2f}%\")" 2>/dev/null || echo "?")
    echo "  ✅ ${train_type^^} step=${step}  acc=$acc"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: Base Model Accuracy
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "【Phase 0】Base Model Accuracy（GPU${GPU_LIST[0]}）"
echo "============================================================"

BASE_RESULT_JSON="${EVAL_DIR}/eval_result_gsm8k_base_noCoT.json"
BASE_RAW_JSON="${EVAL_DIR}/raw_base_noCoT.json"
BASE_LOG="${LOG_DIR}/eval_gsm8k_base_noCoT.log"

if [[ -f "$BASE_RESULT_JSON" ]]; then
    echo "  ⏭  跳过(已有): base model"
else
    echo "  🚀 Base Model（无 LoRA adapter）"
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python "${DGO_ROOT}/vllm_inference.py" \
        --model_name "$BASE_MODEL" \
        --dataset gsm8k \
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
        --output_file "$BASE_RAW_JSON" \
        --stop '</answer>' \
        --stop $'\nQ:' \
        --stop $'\n\nQ:' \
        > "$BASE_LOG" 2>&1

    python3 - <<PYEOF >> "$BASE_LOG" 2>&1
import json, re

def extract_boxed(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    s = m.group(1) if m else text
    idx = s.rfind(r'\\boxed{')
    if idx == -1: return None
    d = 0
    for i in range(s.index('{', idx), len(s)):
        if s[i] == '{': d += 1
        elif s[i] == '}':
            d -= 1
            if d == 0: return s[s.index('{', idx)+1:i].strip()
    return None

def norm(s):
    if s is None: return None
    s = s.replace(',','').strip().rstrip('.')
    try:
        f = float(s); return str(int(f)) if f==int(f) else str(f)
    except: return s

def extract_gt(text):
    if text is None: return None
    m = re.search(r'####\s*([\d,\.]+)', text)
    return m.group(1).replace(',','').strip() if m else text.strip()

with open('${BASE_RAW_JSON}') as f:
    data = json.load(f)

total, correct, no_answer = len(data), 0, 0
details = []
for item in data:
    response = item['generated_texts'][0]
    pred = norm(extract_boxed(response))
    gt   = norm(extract_gt(item['ground_truth_answer']))
    ok   = pred is not None and gt is not None and pred == gt
    if ok: correct += 1
    if pred is None: no_answer += 1
    prompt = item['prompt']
    q_start = prompt.rfind('\nQ:')
    question = prompt[q_start+3:].strip() if q_start != -1 else prompt[-300:].strip()
    details.append({
        'question':     question,
        'response':     response,
        'predicted':    pred,
        'ground_truth': item['ground_truth_answer'],
        'gt_normalized': gt,
        'correct':      ok,
    })

accuracy = correct / total * 100 if total > 0 else 0
result = {
    'train_type': 'base', 'step': 0, 'model': 'qwen',
    'dataset': 'gsm8k', 'split': 'test', 'total': total,
    'correct': correct, 'accuracy': accuracy, 'no_answer': no_answer,
    'details': details,
}
with open('${BASE_RESULT_JSON}','w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f'✅ base model: {accuracy:.2f}% ({correct}/{total}) no_ans={no_answer}')
PYEOF

    acc=$(python3 -c "import json; d=json.load(open('${BASE_RESULT_JSON}')); print(f\"{d['accuracy']:.2f}%\")" 2>/dev/null || echo "?")
    echo "  ✅ Base Model  acc=$acc"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Accuracy 评测（单 GPU 串行，逐个 checkpoint）
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "【Phase 1】Accuracy 评测（${N_GPU}-GPU 并行，每轮 ${N_GPU} 个 checkpoint）"
echo "============================================================"
echo ""

# 每轮同时启动 N_GPU 个 checkpoint，等全部完成后进入下一轮
batch_idx=0
while [ $batch_idx -lt $TOTAL ]; do
    pids=()
    for (( slot=0; slot<N_GPU && batch_idx+slot<TOTAL; slot++ )); do
        entry="${ALL_CKPTS[$((batch_idx + slot))]}"
        IFS=':' read -r tt ckpt step <<< "$entry"
        gpu_id="${GPU_LIST[$slot]}"
        run_acc "$gpu_id" "$tt" "$ckpt" "$step" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    batch_idx=$((batch_idx + N_GPU))
    echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: 清洗预测（生成 _clean 版本）
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "【Phase 2】清洗预测 (fix_gsm8k_predictions.py → *_clean.json)"
echo "============================================================"

# 收集本轮评测生成的所有 *_noCoT.json，加上 base 的（不在 checkpoint 目录里）
CLEAN_FILES=()
for entry in "${ALL_CKPTS[@]}"; do
    IFS=':' read -r tt ckpt step <<< "$entry"
    rf="${ckpt}/eval_result_gsm8k_noCoT.json"
    [[ -f "$rf" ]] && CLEAN_FILES+=("$rf")
done
[[ -f "$BASE_RESULT_JSON" ]] && CLEAN_FILES+=("$BASE_RESULT_JSON")

if [[ ${#CLEAN_FILES[@]} -gt 0 ]]; then
    python3 "${DGO_ROOT}/fix_gsm8k_predictions.py" "${CLEAN_FILES[@]}" || true
else
    echo "⚠️  无可清洗的结果文件"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Accuracy 汇总（优先展示 _clean 版本）
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Accuracy 汇总（cleaned）"
echo "============================================================"
# Base
if [[ -f "${BASE_RESULT_JSON%.json}_clean.json" ]]; then
    acc=$(python3 -c "import json; d=json.load(open('${BASE_RESULT_JSON%.json}_clean.json')); print(f\"{d['accuracy']:.2f}%\")" 2>/dev/null || echo "?")
    printf "  %-5s %-10s  %s\n" "BASE" "" "$acc"
elif [[ -f "$BASE_RESULT_JSON" ]]; then
    acc=$(python3 -c "import json; d=json.load(open('$BASE_RESULT_JSON')); print(f\"{d['accuracy']:.2f}%\")" 2>/dev/null || echo "?")
    printf "  %-5s %-10s  %s  (raw)\n" "BASE" "" "$acc"
fi
# Checkpoints
for entry in "${ALL_CKPTS[@]}"; do
    IFS=':' read -r tt ckpt step <<< "$entry"
    rf_clean="${ckpt}/eval_result_gsm8k_noCoT_clean.json"
    rf_raw="${ckpt}/eval_result_gsm8k_noCoT.json"
    if [[ -f "$rf_clean" ]]; then
        acc=$(python3 -c "import json; d=json.load(open('$rf_clean')); print(f\"{d['accuracy']:.2f}%\")" 2>/dev/null || echo "?")
        printf "  %-5s step=%4s  %s\n" "${tt^^}" "$step" "$acc"
    elif [[ -f "$rf_raw" ]]; then
        acc=$(python3 -c "import json; d=json.load(open('$rf_raw')); print(f\"{d['accuracy']:.2f}%\")" 2>/dev/null || echo "?")
        printf "  %-5s step=%4s  %s  (raw)\n" "${tt^^}" "$step" "$acc"
    fi
done

echo ""
echo "============================================================"
echo " ✅ GSM8K no-CoT 评测 + 清洗完成！"
echo " 原始结果: *_noCoT.json"
echo " 清洗结果: *_noCoT_clean.json"
echo " 结果目录: $EVAL_DIR"
echo " 结束时间: $(date)"
echo "============================================================"
