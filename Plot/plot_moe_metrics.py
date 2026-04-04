#!/usr/bin/env python3
"""
MoE Metrics 绘图脚本

用法:
    # 自动扫描所有 moe_metrics.json 并绘图
    python Plot/plot_moe_metrics.py

    # 指定 outputs 目录
    python Plot/plot_moe_metrics.py --output_root /wutailin/DGO/outputs

    # 单独绘制某个实验
    python Plot/plot_moe_metrics.py --files /path/to/moe_metrics.json

    # 多实验对比图
    python Plot/plot_moe_metrics.py --compare

输出:
    - 单实验图: 保存到对应实验的 moe_logs/ 目录
    - 对比图: 保存到 Plot/ 目录
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 中文支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 配置
# =============================================================================
# 要绘制的指标及其显示配置
METRIC_GROUPS = {
    'load_balance': {
        'title': 'Load Balance',
        'metrics': ['load_cv', 'collapse_rate'],
        'ylabel': 'Value',
    },
    'routing': {
        'title': 'Routing',
        'metrics': ['routing_entropy', 'max_load'],
        'ylabel': 'Value',
    },
    'maxvio': {
        'title': 'Max Violation',
        'metrics': ['maxvio_min_global', 'maxvio_max_global', 'maxvio_min_batch', 'maxvio_max_batch'],
        'ylabel': 'Value',
    },
    'aux_loss': {
        'title': 'Auxiliary Loss',
        'metrics': ['aux_loss'],
        'ylabel': 'Loss',
    },
}

# 每个指标的显示名称、颜色、方向（lower_better=True 表示越低越好）
METRIC_STYLE = {
    'load_cv':           {'label': 'Load CV',            'color': '#e74c3c', 'ls': '-',  'lower_better': True},
    'collapse_rate':     {'label': 'Collapse Rate',      'color': '#3498db', 'ls': '-',  'lower_better': True},
    'routing_entropy':   {'label': 'Routing Entropy',    'color': '#2ecc71', 'ls': '-',  'lower_better': False},  # 越高越均匀
    'max_load':          {'label': 'Max Load',           'color': '#f39c12', 'ls': '-',  'lower_better': True},
    'maxvio_min_batch':  {'label': 'MaxVio Min (batch)', 'color': '#9b59b6', 'ls': '--', 'lower_better': False},  # 越高越好（min 接近 1 = 均匀）
    'maxvio_max_batch':  {'label': 'MaxVio Max (batch)', 'color': '#e74c3c', 'ls': '--', 'lower_better': True},   # 越低越好（max 接近 1 = 均匀）
    'maxvio_min_global': {'label': 'MaxVio Min (global)','color': '#9b59b6', 'ls': '-',  'lower_better': False},  # 越高越好
    'maxvio_max_global': {'label': 'MaxVio Max (global)','color': '#e74c3c', 'ls': '-',  'lower_better': True},   # 越低越好
    'aux_loss':          {'label': 'Aux Loss',           'color': '#1abc9c', 'ls': '-',  'lower_better': True},
}

def _direction_arrow(metric_name):
    """返回指标方向箭头"""
    style = METRIC_STYLE.get(metric_name, {})
    if style.get('lower_better', True):
        return '\u2193'  # ↓ 越低越好
    else:
        return '\u2191'  # ↑ 越高越好

# 训练类型的颜色
TRAIN_TYPE_COLORS = {
    'SFT':  '#3498db',
    'GRPO': '#e74c3c',
    'DGO':  '#2ecc71',
}


def load_metrics(filepath):
    """加载 moe_metrics.json"""
    with open(filepath) as f:
        return json.load(f)


def parse_experiment_info(filepath):
    """从路径解析实验信息"""
    parts = str(filepath).split('/')
    info = {'path': filepath, 'model': 'unknown', 'dataset': 'unknown', 'train_type': 'unknown', 'version': ''}

    for p in parts:
        if 'swift_sft' in p:
            info['train_type'] = 'SFT'
        elif 'swift_grpo' in p:
            info['train_type'] = 'GRPO'
        elif 'swift_dgo' in p:
            info['train_type'] = 'DGO'

    # 找 model_dataset 目录名
    for i, p in enumerate(parts):
        if p.startswith('v') and '-' in p and len(p) > 5:
            # 前一个目录是 model_dataset
            if i > 0:
                model_dataset = parts[i - 1]
                # olmoe_bigmath / olmoe_instruct_bigmath
                if '_bigmath' in model_dataset:
                    info['model'] = model_dataset.replace('_bigmath', '')
                    info['dataset'] = 'bigmath'
                elif '_gsm8k' in model_dataset:
                    info['model'] = model_dataset.replace('_gsm8k', '')
                    info['dataset'] = 'gsm8k'
                elif '_math' in model_dataset:
                    info['model'] = model_dataset.replace('_math', '')
                    info['dataset'] = 'math'
                else:
                    info['model'] = model_dataset
            info['version'] = p
            break

    info['label'] = f"{info['model']}_{info['train_type']}"
    info['title'] = f"{info['model']} - {info['train_type']} ({info['dataset']})"
    return info


def plot_single_experiment(metrics, info, save_dir):
    """为单个实验绘制所有指标"""
    steps = metrics.get('step', [])
    if not steps:
        print(f"  跳过（无数据）: {info['path']}")
        return

    os.makedirs(save_dir, exist_ok=True)

    # 1. 分组子图（一张大图）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"MoE Metrics: {info['title']}", fontsize=14, fontweight='bold')

    for ax, (group_key, group_cfg) in zip(axes.flat, METRIC_GROUPS.items()):
        for metric_name in group_cfg['metrics']:
            if metric_name in metrics and metrics[metric_name]:
                style = METRIC_STYLE.get(metric_name, {})
                arrow = _direction_arrow(metric_name)
                ax.plot(steps, metrics[metric_name],
                        label=f"{style.get('label', metric_name)} {arrow}",
                        color=style.get('color', None),
                        linestyle=style.get('ls', '-'),
                        linewidth=1.5, alpha=0.85)
        ax.set_title(group_cfg['title'], fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel(group_cfg['ylabel'])
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'moe_metrics_all.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {save_path}")

    # 2. 每个指标单独一张图
    for metric_name in ['load_cv', 'collapse_rate', 'routing_entropy', 'max_load', 'aux_loss',
                         'maxvio_min_batch', 'maxvio_max_batch',
                         'maxvio_min_global', 'maxvio_max_global']:
        if metric_name not in metrics or not metrics[metric_name]:
            continue
        style = METRIC_STYLE.get(metric_name, {})
        arrow = _direction_arrow(metric_name)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, metrics[metric_name],
                color=style.get('color', '#333'),
                linewidth=1.5)
        ax.set_title(f"{style.get('label', metric_name)} {arrow} - {info['title']}", fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel(style.get('label', metric_name))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'moe_{metric_name}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  保存: {save_dir}/moe_*.png ({len(steps)} steps)")


def plot_comparison(all_experiments, save_dir):
    """多实验对比图"""
    if len(all_experiments) < 2:
        print("对比图需要至少 2 个实验")
        return

    os.makedirs(save_dir, exist_ok=True)

    # 按 (model, dataset) 分组
    groups = defaultdict(list)
    for exp in all_experiments:
        key = (exp['info']['model'], exp['info']['dataset'])
        groups[key].append(exp)

    for (model, dataset), exps in groups.items():
        if len(exps) < 2:
            continue

        # 对比的关键指标
        compare_metrics = ['load_cv', 'collapse_rate', 'routing_entropy',
                          'maxvio_max_global', 'maxvio_min_global',
                          'maxvio_max_batch', 'maxvio_min_batch',
                          'max_load', 'aux_loss']

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle(f"MoE Metrics Comparison: {model} ({dataset})", fontsize=14, fontweight='bold')

        for ax, metric_name in zip(axes.flat, compare_metrics):
            for exp in exps:
                metrics = exp['metrics']
                info = exp['info']
                steps = metrics.get('step', [])
                if metric_name not in metrics or not metrics[metric_name]:
                    continue
                color = TRAIN_TYPE_COLORS.get(info['train_type'], '#333')
                ax.plot(steps, metrics[metric_name],
                        label=info['train_type'],
                        color=color,
                        linewidth=1.5, alpha=0.85)

            style = METRIC_STYLE.get(metric_name, {})
            arrow = _direction_arrow(metric_name)
            ax.set_title(f"{style.get('label', metric_name)} {arrow}", fontsize=11)
            ax.set_xlabel('Step')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'compare_{model}_{dataset}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  对比图: {save_path}")


def find_all_metrics(output_root):
    """递归扫描所有 moe_metrics.json"""
    results = []
    for root, dirs, files in os.walk(output_root):
        if 'moe_metrics.json' in files:
            filepath = os.path.join(root, 'moe_metrics.json')
            results.append(filepath)
    return sorted(results)


def main():
    parser = argparse.ArgumentParser(description='MoE Metrics 绘图脚本')
    parser.add_argument('--output_root', type=str, default=None,
                        help='outputs 根目录，自动扫描所有 moe_metrics.json')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                        help='手动指定 moe_metrics.json 文件路径')
    parser.add_argument('--compare', action='store_true',
                        help='生成多实验对比图')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='对比图保存目录（默认 Plot/）')
    args = parser.parse_args()

    # 确定 Plot 目录
    script_dir = Path(__file__).parent
    plot_dir = args.save_dir or str(script_dir)

    # 查找所有 metrics 文件
    if args.files:
        metric_files = args.files
    else:
        # 自动扫描
        output_root = args.output_root
        if output_root is None:
            # 尝试多个可能的路径
            candidates = [
                '/wutailin/DGO/outputs',
                str(script_dir.parent / 'outputs'),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    output_root = c
                    break
            # 也检查符号链接
            symlink = str(script_dir.parent / 'outputs')
            if os.path.islink(symlink):
                output_root = os.path.realpath(symlink)

        if output_root is None or not os.path.isdir(output_root):
            print(f"❌ 找不到 outputs 目录，请用 --output_root 指定")
            sys.exit(1)

        metric_files = find_all_metrics(output_root)

    if not metric_files:
        print("❌ 未找到任何 moe_metrics.json")
        sys.exit(1)

    print(f"找到 {len(metric_files)} 个 MoE metrics 文件:")
    for f in metric_files:
        print(f"  {f}")
    print()

    # 加载并绘制
    all_experiments = []
    for filepath in metric_files:
        metrics = load_metrics(filepath)
        info = parse_experiment_info(filepath)
        print(f"[{info['train_type']}] {info['title']}")

        # 单实验图保存到对应的 moe_logs 目录
        save_dir = os.path.dirname(filepath)
        plot_single_experiment(metrics, info, save_dir)
        all_experiments.append({'metrics': metrics, 'info': info})

    # 对比图
    if args.compare or len(all_experiments) >= 2:
        print(f"\n生成对比图...")
        plot_comparison(all_experiments, plot_dir)

    print(f"\n✅ 绘图完成")


if __name__ == '__main__':
    main()
