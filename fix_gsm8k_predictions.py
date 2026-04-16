#!/usr/bin/env python3
"""
fix_gsm8k_predictions.py
GSM8K eval_result_gsm8k.json 答案规范化修复工具

修复两类问题：
  1. predicted="$18" 但 gt="18"   → 去掉 $ 和千位逗号后重新匹配
  2. predicted=null               → 从 response 末尾提取最后数字

用法：
  # 修复指定文件
  python fix_gsm8k_predictions.py path/to/eval_result_gsm8k.json

  # 扫描多个目录下所有 eval_result_gsm8k.json
  python fix_gsm8k_predictions.py --scan_dirs dir1 dir2 dir3

  # 只预览不修改
  python fix_gsm8k_predictions.py --scan_dirs dir1 --dry_run

  # 同时打印每条修改详情
  python fix_gsm8k_predictions.py --scan_dirs dir1 --verbose
"""

import json, re, sys, argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 提取工具函数
# ─────────────────────────────────────────────────────────────────────────────

def norm(s):
    """归一化数字字符串：去掉 $、千位逗号、尾部句点，统一为最简数字字符串"""
    if s is None:
        return None
    s = str(s).strip().lstrip('$').replace(',', '').rstrip('.')
    try:
        f = float(s)
        if f != f or f == float('inf') or f == float('-inf'):  # nan / inf
            return s.strip() or None
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return s.strip() or None


def extract_boxed(text):
    """从 <answer>\\boxed{...}</answer> 提取，与评测脚本逻辑一致"""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    s = m.group(1) if m else text
    idx = s.rfind(r'\boxed{')
    if idx == -1:
        return None
    depth = 0
    brace_start = s.index('{', idx)
    for i in range(brace_start, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return s[brace_start + 1:i].strip()
    return None


def extract_from_answer_tag(text):
    """从 <answer>...</answer> 标签内提取纯数字（无 boxed 时的 fallback）"""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not m:
        return None
    content = m.group(1).strip()
    # 去掉 \boxed 等 LaTeX 残留，直接找数字
    nums = re.findall(r'\$?([\d,]+(?:\.\d+)?)', content)
    return norm(nums[-1]) if nums else None


def extract_last_number(text):
    """
    从 response 末尾提取最后出现的数字。
    优先匹配常见句式：
      - "The answer is: 7425"
      - "= $57500"
      - "<<1150*50=57500>>"
      - 末尾纯数字
    """
    # 取最后 5 行作为主要搜索区域
    last_lines = '\n'.join(text.strip().splitlines()[-5:])

    patterns = [
        r'[Tt]he answer is[:\s]+\$?([\d,]+(?:\.\d+)?)',
        r'answer[:\s]+\$?([\d,]+(?:\.\d+)?)',
        r'=\s*\$?([\d,]+(?:\.\d+)?)\s*[.>)\]]*\s*$',
        r'<<[^>]+=\s*([\d,]+(?:\.\d+)?)>>',   # GSM8K calculator 格式
        r'\$?([\d,]+(?:\.\d+)?)\s*[.]\s*$',
        r'\$?([\d,]+(?:\.\d+)?)\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, last_lines)
        if m:
            v = norm(m.group(1))
            if v:
                return v

    # 全文最后一个数字（最后保底）
    all_nums = re.findall(r'\$?([\d,]+(?:\.\d+)?)', text)
    return norm(all_nums[-1]) if all_nums else None


def better_extract(item):
    """
    对单条 detail 尝试更好的提取。
    返回 (new_predicted, method_name)
    method_name 说明提取来源，便于统计分析。
    """
    pred     = item.get('predicted')
    response = item.get('response', '')

    # 1. 已有 predicted：重新归一化（处理 $X、千位逗号等）
    if pred is not None:
        normalized = norm(pred)
        if normalized and normalized != pred:
            return normalized, 'normalize_existing'
        return pred, 'unchanged'  # 没变化

    # 2. predicted=null：重新尝试 boxed 提取（防止原脚本 bug）
    boxed = extract_boxed(response)
    if boxed:
        return norm(boxed), 'reextract_boxed'

    # 3. 从 <answer> 标签提取（无 boxed 格式）
    from_tag = extract_from_answer_tag(response)
    if from_tag:
        return from_tag, 'answer_tag'

    # 4. 从 response 末尾提取最后数字
    last_num = extract_last_number(response)
    if last_num:
        return last_num, 'last_number'

    return None, 'failed'


# ─────────────────────────────────────────────────────────────────────────────
# 主处理逻辑
# ─────────────────────────────────────────────────────────────────────────────

def fix_file(path, dry_run=False, verbose=False):
    path = Path(path)
    # 输出到同目录下的 _clean 版本，保留原始文件不变
    out_path = path.parent / path.name.replace('.json', '_clean.json')
    with open(path) as f:
        data = json.load(f)

    details = data.get('details')
    if not details:
        print(f"  ⚠️  跳过（无 details 字段）: {path}")
        return 0

    old_correct  = data['correct']
    old_accuracy = data['accuracy']

    changed       = 0
    newly_correct = 0
    method_counts = {}

    for item in details:
        if item.get('correct'):
            continue  # 已正确，跳过

        gt = item.get('gt_normalized')
        if not gt:
            continue

        new_pred, method = better_extract(item)

        if method == 'unchanged':
            continue  # 无变化

        old_pred      = item.get('predicted')
        is_correct    = (new_pred is not None and new_pred == gt)

        item['predicted'] = new_pred
        item['correct']   = is_correct
        method_counts[method] = method_counts.get(method, 0) + 1
        changed += 1

        if is_correct:
            newly_correct += 1
            if verbose:
                q = item.get('question', '')[:60].replace('\n', ' ')
                print(f"    ✅ [{method}] pred: {repr(old_pred)} → {repr(new_pred)}  gt={gt}")
                print(f"       Q: {q}")
        else:
            if verbose and old_pred is None:
                # null → 提取到了但仍然答错，打印出来看看
                print(f"    ⚠️  [{method}] pred: None → {repr(new_pred)}  gt={gt}  (still wrong)")

    # 重新计算汇总统计
    total     = len(details)
    correct   = sum(1 for x in details if x.get('correct'))
    no_answer = sum(1 for x in details if x.get('predicted') is None)
    accuracy  = correct / total * 100 if total > 0 else 0

    data['correct']   = correct
    data['accuracy']  = accuracy
    data['no_answer'] = no_answer

    delta = accuracy - old_accuracy
    ckpt_label = f"{path.parent.parent.parent.name}/{path.parent.name}"
    print(f"  {ckpt_label}: {old_accuracy:.2f}% → {accuracy:.2f}% "
          f"(+{delta:.2f}%)  changed={changed}  new_correct={newly_correct}  "
          f"methods={method_counts}")
    if not dry_run:
        print(f"    → 写入: {out_path.name}")

    if not dry_run:
        with open(out_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return newly_correct


def main():
    parser = argparse.ArgumentParser(
        description='修复 GSM8K eval_result_gsm8k.json 中的答案提取问题',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('files', nargs='*',
                        help='直接指定 JSON 文件路径')
    parser.add_argument('--scan_dirs', nargs='+', metavar='DIR',
                        help='扫描目录下所有匹配 --pattern 的 JSON（递归）')
    parser.add_argument('--pattern', default='eval_result_gsm8k.json',
                        help='--scan_dirs 时的文件名模式 (默认 eval_result_gsm8k.json；'
                             'no-CoT 评测请用 eval_result_gsm8k_noCoT.json)')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印统计，不修改文件')
    parser.add_argument('--verbose', action='store_true',
                        help='打印每条修改的详情')
    args = parser.parse_args()

    files = list(args.files or [])

    if args.scan_dirs:
        for d in args.scan_dirs:
            for p in sorted(Path(d).rglob(args.pattern)):
                if 'merged' not in str(p) and '_clean' not in p.name:
                    files.append(str(p))

    if not files:
        parser.print_help()
        sys.exit(1)

    mode = '（dry run，不修改文件）' if args.dry_run else '（原地修改）'
    print(f"处理 {len(files)} 个文件 {mode}\n")

    total_new_correct = 0
    for f in files:
        r = fix_file(f, dry_run=args.dry_run, verbose=args.verbose)
        total_new_correct += (r or 0)

    print(f"\n{'='*60}")
    print(f"  总计新增正确答案: {total_new_correct}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
