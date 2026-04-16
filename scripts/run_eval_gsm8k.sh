#!/bin/bash
# =============================================================================
# 在 GSM8K 测试集上评测 qwen bigmath 所有 checkpoints (SFT / GRPO / DGO)
# 2-GPU 并行，结果保存到各 checkpoint 的 eval_result_gsm8k.json
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

GPU1="${GPU1:-2}"
GPU2="${GPU2:-3}"

BASE_MODEL="${QWEN_MOE_MODEL}"
EVAL_DIR="${DGO_STORAGE}/eval/qwen_gsm8k_all"
LOG_DIR="${DGO_LOGS}/eval"
PLOT_DIR="${DGO_ROOT}/Plot"

mkdir -p "$EVAL_DIR" "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 列表
# ─────────────────────────────────────────────────────────────────────────────
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

TOTAL=${#ALL_CKPTS[@]}

LOG_FILE="${LOG_DIR}/eval_qwen_gsm8k_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " qwen GSM8K 批量评测（2-GPU 并行）"
echo " GPU1=$GPU1  GPU2=$GPU2"
echo " 总 checkpoints: $TOTAL"
echo " 日志: $LOG_FILE"
echo "============================================================"
echo "开始时间: $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# 单个 checkpoint 推理函数
# ─────────────────────────────────────────────────────────────────────────────
run_one_on_gpu() {
    local gpu_id="$1"
    local train_type="$2"
    local ckpt_dir="$3"
    local step="$4"

    local result_json="${ckpt_dir}/eval_result_gsm8k.json"
    local raw_json="${EVAL_DIR}/raw_${train_type}_step${step}.json"
    local ckpt_log="${LOG_DIR}/eval_gsm8k_${train_type}_step${step}.log"

    # SFT / GRPO / DGO 统一使用 no-CoT prompt（仅 <answer>\boxed{}），研究约束：直接答题不允许思考
    local sys_prompt_arg=""

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

    # 计算准确率（GSM8K ground truth 是 "#### 42" 格式）
    python3 - <<PYEOF >> "$ckpt_log" 2>&1
import json, re, sys

def extract_boxed(text):
    """从模型输出提取 \\boxed{} 内容"""
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

def extract_gsm8k_gt(text):
    """从 GSM8K ground truth 提取答案（#### 42 格式）"""
    if text is None:
        return None
    m = re.search(r'####\s*([\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    return text.strip()

def normalize_num(s):
    """归一化数字字符串"""
    if s is None:
        return None
    s = s.replace(',', '').strip().rstrip('.')
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except:
        return s

with open('${raw_json}') as f:
    data = json.load(f)

total, correct, no_answer = len(data), 0, 0
details = []
for item in data:
    completion = item['generated_texts'][0]
    gt_raw = item['ground_truth_answer']
    pred = normalize_num(extract_boxed(completion))
    gt = normalize_num(extract_gsm8k_gt(gt_raw))
    ok = (pred is not None) and (gt is not None) and (pred == gt)
    if ok: correct += 1
    if pred is None: no_answer += 1
    details.append({'ground_truth': str(gt_raw)[:60], 'predicted': pred, 'correct': ok})

accuracy = correct / total * 100 if total > 0 else 0
result = {
    'train_type': '${train_type}',
    'step': ${step},
    'model': 'qwen',
    'dataset': 'gsm8k',
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
# 2-GPU 并行主循环
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo " 开始 2-GPU 并行推理..."
echo ""

idx=0
while [ $idx -lt $TOTAL ]; do
    entry1="${ALL_CKPTS[$idx]}"
    entry2="${ALL_CKPTS[$((idx+1))]:-}"

    IFS=':' read -r tt1 ckpt1 step1 <<< "$entry1"

    if [[ -f "${ckpt1}/eval_result_gsm8k.json" ]]; then
        echo "  [GPU$GPU1] ⏭  已有结果跳过: ${tt1} step=${step1}"
        pid1=""
    else
        run_one_on_gpu "$GPU1" "$tt1" "$ckpt1" "$step1" &
        pid1=$!
    fi

    pid2=""
    if [[ -n "$entry2" ]]; then
        IFS=':' read -r tt2 ckpt2 step2 <<< "$entry2"
        if [[ -f "${ckpt2}/eval_result_gsm8k.json" ]]; then
            echo "  [GPU$GPU2] ⏭  已有结果跳过: ${tt2} step=${step2}"
        else
            run_one_on_gpu "$GPU2" "$tt2" "$ckpt2" "$step2" &
            pid2=$!
        fi
    fi

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
    rf="${ckpt}/eval_result_gsm8k.json"
    if [[ -f "$rf" ]]; then
        acc=$(python3 -c "import json; d=json.load(open('$rf')); print(f\"{d['accuracy']:.2f}%\")" 2>/dev/null || echo "?")
        printf "  %-5s step=%4s  %s\n" "${tt^^}" "$step" "$acc"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 绘图（复用 plot_accuracy.py，但扫描 eval_result_gsm8k.json）
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " 绘图..."
echo "============================================================"
python3 - <<'PYEOF'
import json, os, glob
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

STYLE = {
    'sft':  {'color': '#3498db', 'marker': 'o', 'label': 'SFT'},
    'grpo': {'color': '#e74c3c', 'marker': 's', 'label': 'GRPO'},
    'dgo':  {'color': '#2ecc71', 'marker': '^', 'label': 'DGO'},
}

dirs = [
    '/usr/storage/fwan/DGO/outputs/swift_sft/qwen_bigmath/v5-20260328-173809',
    '/usr/storage/fwan/DGO/outputs/swift_grpo/qwen_bigmath/v2-20260328-224328',
    '/usr/storage/fwan/DGO/outputs/swift_dgo/qwen_bigmath/v3-20260405-155724',
]

results = []
for d in dirs:
    for ckpt in sorted(glob.glob(os.path.join(d, 'checkpoint-[0-9]*'))):
        if 'merged' in ckpt:
            continue
        rf = os.path.join(ckpt, 'eval_result_gsm8k.json')
        if os.path.exists(rf):
            with open(rf) as f:
                results.append(json.load(f))

groups = defaultdict(list)
for r in results:
    groups[r['train_type']].append(r)
for tt in groups:
    groups[tt].sort(key=lambda x: x['step'])

save_dir = '/usr/commondata/public/hf_hub/cc/DGOforMoE/Plot'

# 曲线对比图
fig, ax = plt.subplots(figsize=(10, 6))
for tt, items in groups.items():
    style = STYLE.get(tt, {'color': '#333', 'marker': 'x', 'label': tt.upper()})
    steps = [r['step'] for r in items]
    accs = [r['accuracy'] for r in items]
    ax.plot(steps, accs, color=style['color'], marker=style['marker'],
            label=style['label'], linewidth=2, markersize=6, alpha=0.9)
    if accs:
        ax.annotate(f'{accs[-1]:.1f}%', xy=(steps[-1], accs[-1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color=style['color'])

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('GSM8K Test Accuracy: SFT vs GRPO vs DGO (Qwen)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
path = os.path.join(save_dir, 'accuracy_gsm8k_comparison.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'保存: {path}')

# 汇总表
print('\n' + '=' * 55)
print('  GSM8K Accuracy Summary')
print('=' * 55)
for tt in ['sft', 'grpo', 'dgo']:
    items = groups.get(tt, [])
    if items:
        label = STYLE.get(tt, {}).get('label', tt.upper())
        best = max(items, key=lambda x: x['accuracy'])
        print(f'  {label}: best={best["accuracy"]:.2f}% (step {best["step"]}), '
              f'final={items[-1]["accuracy"]:.2f}% (step {items[-1]["step"]})')
print('=' * 55)
PYEOF

echo ""
echo "============================================================"
echo " ✅ 全部完成！"
echo " 图片: ${PLOT_DIR}/accuracy_gsm8k_comparison.png"
echo " 结束时间: $(date)"
echo "============================================================"
