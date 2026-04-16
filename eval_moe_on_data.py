#!/usr/bin/env python3
"""
在指定数据集上评测 MoE 路由健康指标（9个指标）
直接复用 ms-swift 的 MoEMonitorCallback，保证与训练时指标完全一致。

用法:
    python eval_moe_on_data.py \
        --base_model /path/to/base_model \
        --adapters /path/to/sft/checkpoint-1000 \
                   /path/to/grpo/checkpoint-1248 \
                   /path/to/dgo/checkpoint-1252 \
        --labels SFT GRPO DGO \
        --dataset gsm8k \
        --num_samples 500 \
        --output_dir Plot/moe_eval_gsm8k \
        --save_plot Plot/moe_gsm8k_fair_compare.png
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# 环境
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault('HF_HOME', '/root/.cache/huggingface')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

DGO_ROOT = Path(__file__).parent
sys.path.insert(0, str(DGO_ROOT))
# 让 ms-swift 可以被 import
sys.path.insert(0, str(DGO_ROOT / 'ms-swift'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# 直接导入训练时使用的 MoEMonitorCallback
from swift.trainers.moe_callback import MoEMonitorCallback


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────
MATH_SYSTEM = (
    "You are an expert mathematical problem solver. Give your final answer as \\boxed{ANSWER} "
    "inside <answer> tags."
)

def load_eval_prompts(dataset_name: str, num_samples: int, tokenizer) -> List[str]:
    if dataset_name == 'gsm8k':
        ds = load_dataset('openai/gsm8k', 'main', split='test')
        questions = [item['question'] for item in ds]
    elif dataset_name == 'bigmath':
        ds = load_dataset('SynthLabsAI/Big-Math-RL-Verified', split='test')
        questions = [item['problem'] for item in ds]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if 0 < num_samples < len(questions):
        import random; random.seed(42)
        questions = random.sample(questions, num_samples)

    prompts = []
    for q in questions:
        messages = [
            {'role': 'system', 'content': MATH_SYSTEM},
            {'role': 'user',   'content': q},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<|im_start|>system\n{MATH_SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(text)

    print(f"  加载 {len(prompts)} 条 {dataset_name} 提示")
    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# 主评测：直接用 MoEMonitorCallback 的 hook + 指标计算
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_one_checkpoint(
    base_model_path: str,
    adapter_path: str,
    prompts: List[str],
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
) -> Dict[str, float]:
    if adapter_path == "BASE":
        print(f"\n  加载: {base_model_path} (base model, no adapter)")
    else:
        print(f"\n  加载: {adapter_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
    )
    if adapter_path != "BASE":
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # 实例化 callback（model_key="qwen"，与训练时一致）
    cb = MoEMonitorCallback(model_key='qwen', enabled=True, verbose=False)

    # 手动触发 on_train_begin 里的初始化逻辑
    cb.moe_config, cb.detected_model = cb._detect_and_configure(model)
    if cb.moe_config is None:
        raise RuntimeError("无法检测 MoE 架构")
    print(f"  检测到模型: {cb.detected_model}, "
          f"experts={cb.moe_config.num_experts}, topk={cb.moe_config.topk}")

    # 注册 hooks（与训练时完全相同的逻辑）
    cb._register_hooks(model)

    # 分 batch 跑 forward pass（prefill only，不需要 generate）
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        _ = model(**enc)

        if (i // batch_size + 1) % 20 == 0 or i + batch_size >= len(prompts):
            print(f"    [{min(i+batch_size, len(prompts))}/{len(prompts)}] done")

    # 用 callback 自己的 _compute_metrics()，与训练时完全一致
    metrics = cb._compute_metrics()

    # 清理
    for h in cb.hook_handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(all_results: List[Dict], labels: List[str], save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    METRICS = [
        ('load_cv',           'Load CV ↓',             True),
        ('collapse_rate',     'Collapse Rate ↓',        True),
        ('routing_entropy',   'Routing Entropy ↑',      False),
        ('max_load',          'Max Load ↓',             True),
        ('maxvio_min_batch',  'MaxVio Min (batch) ↑',   False),
        ('maxvio_max_batch',  'MaxVio Max (batch) ↓',   True),
        ('maxvio_min_global', 'MaxVio Min (global) ↑',  False),
        ('maxvio_max_global', 'MaxVio Max (global) ↓',  True),
        ('aux_loss',          'Aux Loss ↓',             True),
    ]

    COLORS = {'SFT': '#3498db', 'GRPO': '#e74c3c', 'DGO': '#2ecc71'}
    colors = [COLORS.get(l, '#888') for l in labels]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        'MoE Routing Metrics on GSM8K Test Set\n(Fair Comparison: Same Data, Final Checkpoints)',
        fontsize=13, fontweight='bold'
    )

    for ax, (key, title, lower_better) in zip(axes.flat, METRICS):
        vals = [r.get(key, 0.0) for r in all_results]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f'{v:.4f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        # 金色边框标注最优值
        best_idx = vals.index(min(vals) if lower_better else max(vals))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(title, fontsize=11)
        ax.set_ylim(bottom=0, top=max(vals) * 1.2 if max(vals) > 0 else 1)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n保存对比图: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',  required=True)
    parser.add_argument('--adapters',    nargs='+', required=True)
    parser.add_argument('--labels',      nargs='+', required=True)
    parser.add_argument('--dataset',     default='gsm8k',
                        choices=['gsm8k', 'bigmath'])
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--max_length',  type=int, default=512)
    parser.add_argument('--output_dir',  default='Plot/moe_eval')
    parser.add_argument('--save_plot',   default=None)
    args = parser.parse_args()

    assert len(args.adapters) == len(args.labels)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载 tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n加载数据集: {args.dataset} ({args.num_samples} samples)")
    prompts = load_eval_prompts(args.dataset, args.num_samples, tokenizer)

    all_results = []
    for adapter, label in zip(args.adapters, args.labels):
        print(f"\n{'='*60}\n 评测: {label}\n{'='*60}")
        metrics = eval_one_checkpoint(
            base_model_path=args.base_model,
            adapter_path=adapter,
            prompts=prompts,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        metrics['label'] = label
        all_results.append(metrics)

        out = os.path.join(args.output_dir, f'moe_eval_{label.lower()}.json')
        with open(out, 'w') as f:
            json.dump(metrics, f, indent=2)

        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<24}: {v:.4f}")

    # 汇总表
    KEYS = ['load_cv','collapse_rate','routing_entropy','max_load',
            'maxvio_min_batch','maxvio_max_batch',
            'maxvio_min_global','maxvio_max_global','aux_loss']
    print(f"\n{'='*60}")
    print(f"  MoE 指标汇总（{args.dataset}, n={args.num_samples}）")
    print(f"{'='*60}")
    print(f"{'Metric':<24}" + "".join(f"{l:>10}" for l in args.labels))
    print("-" * (24 + 10 * len(args.labels)))
    for k in KEYS:
        print(f"{k:<24}" + "".join(f"{r.get(k,0):>10.4f}" for r in all_results))

    save_plot = args.save_plot or os.path.join(args.output_dir, 'moe_compare.png')
    plot_comparison(all_results, args.labels, save_plot)
    print(f"\n✅ 完成！结果: {args.output_dir}")


if __name__ == '__main__':
    main()
