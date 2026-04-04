#!/usr/bin/env python3
"""
Accuracy 曲线绘图脚本 — SFT vs GRPO vs DGO checkpoint 对比

用法:
    # 从汇总文件绘图
    python Plot/plot_accuracy.py

    # 指定汇总文件
    python Plot/plot_accuracy.py --summary /wutailin/DGO/eval/summary_olmoe_bigmath.json

    # 从 checkpoint 目录扫描
    python Plot/plot_accuracy.py --scan_dirs \
        /wutailin/DGO/outputs/swift_sft/olmoe_bigmath/v1-20260322-215620 \
        /wutailin/DGO/outputs/swift_grpo/olmoe_bigmath/v4-20260323-033445 \
        /wutailin/DGO/outputs/swift_dgo/olmoe_bigmath/v0-20260323-072409
"""

import argparse
import json
import os
import glob
from pathlib import Path
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


def load_from_summary(filepath):
    with open(filepath) as f:
        return json.load(f)


def scan_checkpoints(dirs):
    results = []
    for d in dirs:
        for ckpt in sorted(glob.glob(os.path.join(d, 'checkpoint-[0-9]*'))):
            if 'merged' in ckpt:
                continue
            rf = os.path.join(ckpt, 'eval_result.json')
            if os.path.exists(rf):
                with open(rf) as f:
                    results.append(json.load(f))
    return results


def plot_accuracy_curve(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 按 train_type 分组
    groups = defaultdict(list)
    for r in results:
        groups[r['train_type']].append(r)

    for tt in groups:
        groups[tt].sort(key=lambda x: x['step'])

    # ===================== 图1: Accuracy 曲线对比 =====================
    fig, ax = plt.subplots(figsize=(10, 6))

    for tt, items in groups.items():
        style = STYLE.get(tt, {'color': '#333', 'marker': 'x', 'label': tt.upper()})
        steps = [r['step'] for r in items]
        accs = [r['accuracy'] for r in items]
        ax.plot(steps, accs,
                color=style['color'], marker=style['marker'],
                label=style['label'], linewidth=2, markersize=6, alpha=0.9)
        # 标注最终值
        if accs:
            ax.annotate(f'{accs[-1]:.1f}%',
                       xy=(steps[-1], accs[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=style['color'])

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('BigMath Test Accuracy: SFT vs GRPO vs DGO (OLMoE)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'accuracy_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'保存: {save_path}')

    # ===================== 图2: 柱状图（最终 accuracy）=====================
    fig, ax = plt.subplots(figsize=(8, 5))

    final_accs = {}
    for tt, items in groups.items():
        if items:
            final_accs[tt] = items[-1]

    bar_x = list(range(len(final_accs)))
    bar_labels = []
    bar_values = []
    bar_colors = []

    for tt in ['sft', 'grpo', 'dgo']:
        if tt in final_accs:
            r = final_accs[tt]
            style = STYLE.get(tt, {})
            bar_labels.append(f"{style.get('label', tt.upper())}\n(step {r['step']})")
            bar_values.append(r['accuracy'])
            bar_colors.append(style.get('color', '#333'))

    bars = ax.bar(range(len(bar_values)), bar_values, color=bar_colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Final Accuracy Comparison (OLMoE × BigMath)', fontsize=13, fontweight='bold')
    ax.set_ylim(bottom=0, top=max(bar_values) * 1.15 if bar_values else 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'accuracy_final_bar.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'保存: {save_path}')

    # ===================== 图3: 每种方法单独曲线 =====================
    for tt, items in groups.items():
        style = STYLE.get(tt, {'color': '#333', 'marker': 'x', 'label': tt.upper()})
        steps = [r['step'] for r in items]
        accs = [r['accuracy'] for r in items]
        no_ans = [r.get('no_answer', 0) for r in items]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        ax1.plot(steps, accs, color=style['color'], marker=style['marker'],
                 linewidth=2, markersize=6)
        for s, a in zip(steps, accs):
            ax1.annotate(f'{a:.1f}', xy=(s, a), xytext=(0, 8),
                        textcoords='offset points', fontsize=8, ha='center')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title(f'{style["label"]} - Accuracy Curve', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        # No-answer rate
        total = items[0]['total'] if items else 1
        no_ans_pct = [n / total * 100 for n in no_ans]
        ax2.plot(steps, no_ans_pct, color='#e67e22', marker='D',
                 linewidth=2, markersize=5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('No Answer Rate (%)')
        ax2.set_title(f'{style["label"]} - No Answer Rate', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'accuracy_{tt}_detail.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'保存: {save_path}')

    # 打印汇总表格
    print('\n' + '=' * 60)
    print('  Accuracy Summary')
    print('=' * 60)
    for tt in ['sft', 'grpo', 'dgo']:
        items = groups.get(tt, [])
        if items:
            label = STYLE.get(tt, {}).get('label', tt.upper())
            print(f'\n  {label}:')
            for r in items:
                print(f'    step {r["step"]:>5}: {r["accuracy"]:6.2f}%  '
                      f'({r["correct"]}/{r["total"]}, no_ans={r.get("no_answer", "?")})')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='Accuracy 曲线绘图')
    parser.add_argument('--summary', type=str, default=None, help='汇总 JSON 文件')
    parser.add_argument('--scan_dirs', type=str, nargs='+', default=None, help='扫描 checkpoint 目录')
    parser.add_argument('--save_dir', type=str, default=None, help='图片保存目录')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    save_dir = args.save_dir or str(script_dir)

    if args.summary:
        results = load_from_summary(args.summary)
    elif args.scan_dirs:
        results = scan_checkpoints(args.scan_dirs)
    else:
        # 默认: 先找汇总文件，再扫描
        default_summary = '/wutailin/DGO/eval/summary_olmoe_bigmath.json'
        if os.path.exists(default_summary):
            results = load_from_summary(default_summary)
        else:
            default_dirs = [
                '/wutailin/DGO/outputs/swift_sft/olmoe_bigmath/v1-20260322-215620',
                '/wutailin/DGO/outputs/swift_grpo/olmoe_bigmath/v4-20260323-033445',
                '/wutailin/DGO/outputs/swift_dgo/olmoe_bigmath/v0-20260323-072409',
            ]
            results = scan_checkpoints(default_dirs)

    if not results:
        print('❌ 未找到评测结果')
        return

    print(f'加载 {len(results)} 条评测结果')
    plot_accuracy_curve(results, save_dir)
    print('\n✅ 绘图完成')


if __name__ == '__main__':
    main()
