# =============================================================================
# Unified Reward Functions for DGO Training
# =============================================================================
# 使用方法:
#   from reward import get_reward_funcs
#   reward_funcs = get_reward_funcs("gsm8k", style="gfol")  # gfol风格 (有辅助奖励)
#   reward_funcs = get_reward_funcs("gsm8k", style="swift") # ms-swift风格 (仅正确性)
# =============================================================================

import re
import sys
from typing import List, Callable

# =============================================================================
# 导入验证工具
# =============================================================================
import os

# 获取当前文件目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REDIT_PATH = os.path.join(_CURRENT_DIR, 'ReDit')

# MBPP 工具
if _REDIT_PATH not in sys.path:
    sys.path.insert(0, _REDIT_PATH)

try:
    from mbpp_utils import extract_code_from_completion, execute_code_with_tests, validate_python_syntax
    HAS_MBPP_UTILS = True
except ImportError:
    HAS_MBPP_UTILS = False
    print("Warning: mbpp_utils not found, MBPP rewards will not work")

# math_verify (ms-swift 风格)
try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False

# =============================================================================
# 内置数学验证函数 (从 grpo_math.py 提取，避免模块级代码执行)
# =============================================================================

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except (AssertionError, ValueError):
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    """标准化数学字符串以进行比较"""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose=False):
    """检查两个数学表达式是否等价"""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = _strip_string(str(str1))
        ss2 = _strip_string(str(str2))
        if verbose:
            print(f"Comparing: '{ss1}' vs '{ss2}'")
        return ss1 == ss2
    except Exception:
        return str(str1).strip() == str(str2).strip()


def remove_boxed(s):
    """从 \\boxed{} 格式中提取内容"""
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left):]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def last_boxed_only_string(string):
    """提取最后一个 \\boxed{} 内容"""
    if string is None:
        return None
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx:right_brace_idx + 1]


def extract_solution(solution_str):
    """从解答字符串中提取 boxed 答案"""
    if solution_str is None:
        return None
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    return remove_boxed(boxed)


HAS_REDIT = True  # 内置函数始终可用


# =============================================================================
# 辅助函数
# =============================================================================

def extract_xml_answer(text: str) -> str:
    """从 <answer>...</answer> 标签中提取答案"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """从 #### 格式中提取答案 (GSM8K 原始格式)"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


def safe_is_equiv(a: str, b: str) -> bool:
    """安全的数学等价性检查"""
    if HAS_REDIT:
        try:
            return is_equiv(a, b)
        except:
            pass
    # Fallback: 字符串匹配
    return a.strip() == b.strip()


def swift_verify(completion: str, ground_truth: str) -> bool:
    """
    ms-swift 风格的验证 (使用 math_verify)
    支持 <answer>...</answer> 和 \\boxed{} 格式
    """
    if not HAS_MATH_VERIFY:
        return safe_is_equiv(extract_xml_answer(completion), ground_truth)

    try:
        # 尝试从 completion 中解析答案
        answer_parsed = parse(completion)
        gold_parsed = parse(ground_truth)
        return verify(gold_parsed, answer_parsed)
    except:
        # Fallback to is_equiv
        return safe_is_equiv(extract_xml_answer(completion), ground_truth)


# =============================================================================
# MS-Swift 风格 Reward Functions (仅正确性, 1.0/0.0)
# =============================================================================

class SwiftRewards:
    """ms-swift 风格的奖励函数 (仅正确性奖励)"""

    @staticmethod
    def accuracy(prompts, completions, answer, **kwargs) -> List[float]:
        """
        通用正确性奖励 - ms-swift MathAccuracy 风格
        返回: 1.0 (正确) / 0.0 (错误)
        支持: math_verify 或 is_equiv fallback
        """
        rewards = []
        for completion, gt in zip(completions, answer):
            try:
                if swift_verify(completion, gt):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def gsm8k_accuracy(prompts, completions, answer, **kwargs) -> List[float]:
        """GSM8K 正确性 - 提取 XML answer 后验证"""
        rewards = []
        for completion, gt in zip(completions, answer):
            try:
                extracted = extract_xml_answer(completion.strip())
                if safe_is_equiv(extracted, gt):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def math_accuracy(prompts, completions, answer, **kwargs) -> List[float]:
        """MATH 正确性 - 提取 boxed 答案后验证"""
        rewards = []
        for completion, gt in zip(completions, answer):
            try:
                if HAS_REDIT:
                    solution = extract_solution(completion)
                    ground = extract_solution(gt) if gt else gt
                    if is_equiv(solution, ground):
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(1.0 if completion.strip() == gt.strip() else 0.0)
            except:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def mbpp_accuracy(prompts, completions, answer, **kwargs) -> List[float]:
        """MBPP 正确性 - 代码执行验证"""
        if not HAS_MBPP_UTILS:
            return [0.0] * len(completions)

        rewards = []
        for completion, test_cases in zip(completions, answer):
            try:
                code = extract_code_from_completion(completion)
                if not code:
                    rewards.append(0.0)
                    continue

                is_valid, _ = validate_python_syntax(code)
                if not is_valid:
                    rewards.append(0.0)
                    continue

                passed, total, error = execute_code_with_tests(code, test_cases, timeout=2.0)
                if passed == total and total > 0 and not error:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        return rewards


# =============================================================================
# GFOL 风格 Reward Functions (正确性 1.0 + 辅助 0.5 = 总分 1.5)
# =============================================================================

class GFOLRewards:
    """GFOL 风格的奖励函数 (正确性 + 辅助奖励)"""

    # =========================================================================
    # GSM8K (XML 格式)
    # =========================================================================

    @staticmethod
    def gsm8k_correctness(prompts, completions, answer, **kwargs) -> List[float]:
        """GSM8K 正确性奖励 (1.0)"""
        rewards = []
        for completion, gt in zip(completions, answer):
            try:
                extracted = extract_xml_answer(completion.strip())
                if safe_is_equiv(extracted, gt):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def gsm8k_int_reward(completions, **kwargs) -> List[float]:
        """GSM8K 辅助: 答案是整数 (0.1)"""
        responses = [c.strip() for c in completions]
        extracted = [extract_xml_answer(r) for r in responses]
        return [0.1 if r.isdigit() else 0.0 for r in extracted]

    @staticmethod
    def gsm8k_strict_format(completions, **kwargs) -> List[float]:
        """GSM8K 辅助: 严格XML格式 (0.15)"""
        pattern = r"^<reasoning>\n[\s\S]*?\n</reasoning>\n<answer>\n[\s\S]*?</answer>$"
        contents = [c.strip() for c in completions]
        matches = [re.match(pattern, c, re.DOTALL | re.MULTILINE) for c in contents]
        return [0.15 if m else 0.0 for m in matches]

    @staticmethod
    def gsm8k_soft_format(completions, **kwargs) -> List[float]:
        """GSM8K 辅助: 宽松XML格式 (0.15)"""
        pattern = r"^<reasoning>[\s\S]*?</reasoning>[\s\S]*?<answer>[\s\S]*?</answer>$"
        contents = [c.strip() for c in completions]
        matches = [re.match(pattern, c, re.DOTALL | re.MULTILINE) for c in contents]
        return [0.15 if m else 0.0 for m in matches]

    @staticmethod
    def gsm8k_xmlcount(completions, **kwargs) -> List[float]:
        """GSM8K 辅助: XML标签计数 (最高0.1)"""
        def count_xml(text) -> float:
            count = 0.0
            if text.count("<reasoning>\n") == 1: count += 0.025
            if text.count("\n</reasoning>\n") == 1: count += 0.025
            if text.count("\n<answer>\n") == 1: count += 0.025
            if text.count("\n</answer>") == 1:
                count += 0.025
                count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
            return count
        return [count_xml(c.strip()) for c in completions]

    # =========================================================================
    # MATH (\\boxed{} 格式)
    # =========================================================================

    @staticmethod
    def math_correctness(prompts, completions, answer, **kwargs) -> List[float]:
        """MATH 正确性奖励 (1.0)"""
        rewards = []
        for completion, gt in zip(completions, answer):
            try:
                if HAS_REDIT:
                    solution = extract_solution(completion)
                    ground = extract_solution(gt) if gt else gt
                    if is_equiv(solution, ground):
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def math_boxed_format(completions, **kwargs) -> List[float]:
        """MATH 辅助: \\boxed{} 格式 (0.25)"""
        pattern = r"\\boxed\{[^}]+\}"
        contents = [c.strip() for c in completions]
        matches = [re.search(pattern, c) for c in contents]
        return [0.25 if m else 0.0 for m in matches]

    @staticmethod
    def math_reasoning(completions, **kwargs) -> List[float]:
        """MATH 辅助: 包含推理过程 (0.25)"""
        contents = [c.strip() for c in completions]
        return [0.25 if len(c.split('\n')) > 3 else 0.0 for c in contents]

    # =========================================================================
    # MBPP (代码执行)
    # =========================================================================

    @staticmethod
    def mbpp_correctness(prompts, completions, answer, **kwargs) -> List[float]:
        """MBPP 正确性奖励 (1.0)"""
        if not HAS_MBPP_UTILS:
            return [0.0] * len(completions)

        rewards = []
        for completion, test_cases in zip(completions, answer):
            try:
                code = extract_code_from_completion(completion)
                if not code:
                    rewards.append(0.0)
                    continue

                is_valid, _ = validate_python_syntax(code)
                if not is_valid:
                    rewards.append(0.0)
                    continue

                passed, total, error = execute_code_with_tests(code, test_cases, timeout=2.0)
                if passed == total and total > 0 and not error:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def mbpp_syntax(completions, **kwargs) -> List[float]:
        """MBPP 辅助: 语法检查 (0.25)"""
        if not HAS_MBPP_UTILS:
            return [0.0] * len(completions)

        rewards = []
        for completion in completions:
            try:
                code = extract_code_from_completion(completion)
                is_valid, _ = validate_python_syntax(code)
                rewards.append(0.25 if is_valid else 0.0)
            except:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def mbpp_format(completions, **kwargs) -> List[float]:
        """MBPP 辅助: 代码格式 (最高0.25)"""
        if not HAS_MBPP_UTILS:
            return [0.0] * len(completions)

        rewards = []
        for completion in completions:
            try:
                code = extract_code_from_completion(completion)
                score = 0.0
                if 'def ' in code: score += 0.0625
                if '"""' in code or "'''" in code: score += 0.0625
                if 'return ' in code: score += 0.0625
                lines = code.split('\n')
                if any(line.startswith('    ') for line in lines if line.strip()):
                    score += 0.0625
                rewards.append(score)
            except:
                rewards.append(0.0)
        return rewards


# =============================================================================
# 统一接口
# =============================================================================

def get_reward_funcs(
    dataset: str,
    style: str = "gfol",
    include_auxiliary: bool = True
) -> List[Callable]:
    """
    获取指定数据集和风格的奖励函数列表

    Args:
        dataset: 数据集名称 ("gsm8k", "math", "mbpp")
        style: 奖励风格 ("gfol" 或 "swift")
               - "gfol": 正确性 1.0 + 辅助奖励 0.5, 总分 1.5
               - "swift": 仅正确性 1.0, 与 ms-swift 完全一致
        include_auxiliary: 是否包含辅助奖励 (仅 gfol 风格有效)

    Returns:
        奖励函数列表

    Example:
        >>> reward_funcs = get_reward_funcs("gsm8k", style="swift")
        >>> reward_funcs = get_reward_funcs("math", style="gfol", include_auxiliary=True)
    """
    dataset = dataset.lower()
    style = style.lower()

    if style == "swift":
        # ms-swift 风格: 仅正确性奖励
        if dataset == "gsm8k":
            return [SwiftRewards.gsm8k_accuracy]
        elif dataset == "math":
            return [SwiftRewards.math_accuracy]
        elif dataset == "mbpp":
            return [SwiftRewards.mbpp_accuracy]
        else:
            print(f"Warning: Unknown dataset {dataset}, using general accuracy")
            return [SwiftRewards.accuracy]

    elif style == "gfol":
        # GFOL 风格: 正确性 + 辅助奖励
        if dataset == "gsm8k":
            funcs = [GFOLRewards.gsm8k_correctness]
            if include_auxiliary:
                funcs.extend([
                    GFOLRewards.gsm8k_int_reward,
                    GFOLRewards.gsm8k_strict_format,
                    GFOLRewards.gsm8k_soft_format,
                    GFOLRewards.gsm8k_xmlcount,
                ])
            return funcs

        elif dataset == "math":
            funcs = [GFOLRewards.math_correctness]
            if include_auxiliary:
                funcs.extend([
                    GFOLRewards.math_boxed_format,
                    GFOLRewards.math_reasoning,
                ])
            return funcs

        elif dataset == "mbpp":
            funcs = [GFOLRewards.mbpp_correctness]
            if include_auxiliary:
                funcs.extend([
                    GFOLRewards.mbpp_syntax,
                    GFOLRewards.mbpp_format,
                ])
            return funcs

        else:
            print(f"Warning: Unknown dataset {dataset}, using GSM8K as default")
            return get_reward_funcs("gsm8k", style="gfol", include_auxiliary=include_auxiliary)

    else:
        raise ValueError(f"Unknown style: {style}. Use 'gfol' or 'swift'")


def get_max_reward(dataset: str, style: str = "gfol", include_auxiliary: bool = True) -> float:
    """
    获取指定配置的最大可能奖励分数

    Returns:
        最大奖励分数
    """
    if style == "swift":
        return 1.0
    elif style == "gfol":
        if include_auxiliary:
            return 1.5  # 正确性 1.0 + 辅助 0.5
        else:
            return 1.0
    else:
        return 1.0


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reward Functions Test")
    print("=" * 60)

    # 测试数据
    test_prompts = ["What is 2+2?"]
    test_completions = ["<reasoning>\n2+2=4\n</reasoning>\n<answer>\n4\n</answer>"]
    test_answers = ["4"]

    print("\n>>> Testing GSM8K rewards:")

    # Swift 风格
    swift_funcs = get_reward_funcs("gsm8k", style="swift")
    print(f"Swift style: {len(swift_funcs)} function(s), max={get_max_reward('gsm8k', 'swift')}")
    for func in swift_funcs:
        result = func(prompts=test_prompts, completions=test_completions, answer=test_answers)
        print(f"  {func.__name__}: {result}")

    # GFOL 风格
    gfol_funcs = get_reward_funcs("gsm8k", style="gfol")
    print(f"\nGFOL style: {len(gfol_funcs)} function(s), max={get_max_reward('gsm8k', 'gfol')}")
    total = 0.0
    for func in gfol_funcs:
        result = func(prompts=test_prompts, completions=test_completions, answer=test_answers)
        total += result[0]
        print(f"  {func.__name__}: {result}")
    print(f"  Total: {total}")

    print("\n>>> Available configurations:")
    for ds in ["gsm8k", "math", "mbpp"]:
        for st in ["swift", "gfol"]:
            funcs = get_reward_funcs(ds, style=st)
            max_r = get_max_reward(ds, style=st)
            print(f"  {ds}/{st}: {len(funcs)} functions, max_reward={max_r}")

    print("\n" + "=" * 60)
    print("Dependencies status:")
    print(f"  ReDit (is_equiv): {'Available' if HAS_REDIT else 'Not found'}")
    print(f"  math_verify: {'Available' if HAS_MATH_VERIFY else 'Not found'}")
    print(f"  mbpp_utils: {'Available' if HAS_MBPP_UTILS else 'Not found'}")
    print("=" * 60)
