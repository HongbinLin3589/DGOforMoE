#!/usr/bin/env python3
"""
MBPP Execution Utils - lm-evaluation-harness Integration
方案A：复用lm-evaluation-harness的标准实现

Features:
- 代码提取（Markdown和纯代码格式）
- 安全代码执行（multiprocessing隔离）
- 测试用例执行和评估
- 超时保护（2秒）
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import multiprocessing
import io
import re


# =============================================================================
# 代码提取函数
# =============================================================================

def extract_code_from_completion(completion: str) -> str:
    """
    从模型输出中提取Python代码。

    支持的格式：
    1. ```python ... ```
    2. ``` ... ```
    3. 纯代码（无markdown）

    Args:
        completion: 模型生成的完整文本

    Returns:
        提取的Python代码
    """
    # 尝试提取markdown Python代码块
    if "```python" in completion:
        match = re.search(r"```python\s*(.*?)\s*```", completion, re.DOTALL)
        if match:
            return match.group(1).strip()

    # 尝试通用代码块
    if "```" in completion:
        match = re.search(r"```\s*(.*?)\s*```", completion, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # 去除可能的语言标识符
            if code.startswith("python\n"):
                code = code[7:]
            return code

    # 假设全是代码（去除前后空白和注释行）
    lines = completion.split('\n')
    code_lines = []
    in_code = False

    for line in lines:
        # 检测代码开始（def, class, import等）
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'for ', 'while ', 'if ', '@')):
            in_code = True

        if in_code and stripped and not stripped.startswith('#'):
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines).strip()

    # 兜底：返回全部
    return completion.strip()


def extract_code_blocks(text: str) -> str:
    """
    辅助函数：提取代码块（兼容lm-evaluation-harness的命名）
    """
    return extract_code_from_completion(text)


# =============================================================================
# 代码执行引擎
# =============================================================================

def execute_code_with_tests(
    code: str,
    test_cases: List[str],
    timeout: float = 2.0
) -> Tuple[int, int, Optional[str]]:
    """
    在隔离进程中执行代码并运行测试用例。

    Args:
        code: 要执行的Python代码
        test_cases: 测试用例列表（每个是assert语句）
        timeout: 超时时间（秒）

    Returns:
        (passed_count, total_tests, error_message)
        - passed_count: 通过的测试数量
        - total_tests: 总测试数量
        - error_message: 错误信息（如果有）
    """

    def target(queue):
        """在子进程中执行的目标函数"""
        try:
            # 重定向stdout/stderr，保持训练日志清洁
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            # 创建执行环境
            exec_globals = {
                '__builtins__': __builtins__,
                '__name__': '__main__'
            }

            # 执行生成的代码
            exec(code, exec_globals)

            # 执行每个测试用例
            passed = 0
            for test_case in test_cases:
                try:
                    exec(test_case, exec_globals)
                    passed += 1
                except AssertionError:
                    # 测试失败
                    pass
                except Exception as e:
                    # 测试执行错误
                    queue.put((passed, len(test_cases), f"Test execution error: {str(e)}"))
                    return

            # 所有测试通过
            queue.put((passed, len(test_cases), None))

        except SyntaxError as e:
            queue.put((0, len(test_cases), f"SyntaxError: {str(e)}"))
        except Exception as e:
            queue.put((0, len(test_cases), f"RuntimeError: {str(e)}"))

    # 创建队列和子进程
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(queue,))

    # 启动子进程
    process.start()
    process.join(timeout)

    # 检查超时
    if process.is_alive():
        process.terminate()
        process.join()
        return 0, len(test_cases), "Timeout: Code execution exceeded time limit"

    # 检查进程崩溃
    if queue.empty():
        return 0, len(test_cases), "Process crashed without output"

    # 获取结果
    result = queue.get()
    return result


# =============================================================================
# 验证函数
# =============================================================================

def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    快速验证Python语法（不执行）

    Args:
        code: Python代码

    Returns:
        (is_valid, error_message)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def calculate_pass_at_k(results: List[bool], k: int = 1) -> float:
    """
    计算pass@k指标（MBPP标准评估）

    pass@k = 1 - (1-p)^k 其中 p 是单次通过的概率
    简化版本：至少有1个通过即为pass

    Args:
        results: 布尔列表，表示每次采样的pass/fail
        k: 考虑前k个样本

    Returns:
        pass@k的值（0.0-1.0）
    """
    n = len(results)
    if n == 0 or k > n:
        return 0.0

    passes = sum(results[:k])
    return 1.0 if passes > 0 else 0.0


# =============================================================================
# 日志和调试
# =============================================================================

def log_execution_result(
    prompt_id: int,
    passed: int,
    total: int,
    error: Optional[str] = None,
    verbose: bool = False
):
    """记录执行结果（可选）"""
    if verbose:
        status = "✅ PASS" if passed == total else "❌ FAIL"
        print(f"[Prompt {prompt_id}] {status} ({passed}/{total} tests)")
        if error:
            print(f"  Error: {error}")


# =============================================================================
# 测试代码（方案A - lm-evaluation-harness集成）
# =============================================================================

if __name__ == "__main__":
    print("Testing MBPP utils (lm-evaluation-harness integration)...")
    print("=" * 70)

    # 测试用例1：正确的代码
    print("\nTest 1: Correct code")
    code1 = """
def add_two(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""
    tests1 = [
        "assert add_two(3, 4) == 7",
        "assert add_two(0, 0) == 0",
        "assert add_two(-1, 1) == 0"
    ]

    passed, total, error = execute_code_with_tests(code1, tests1)
    print(f"Result: {passed}/{total} passed. Error: {error}")

    # 测试用例2：错误的代码
    print("\nTest 2: Incorrect code")
    code2 = """
def add_two(a, b):
    return a - b  # 错误：应该是 +
"""

    passed, total, error = execute_code_with_tests(code2, tests1)
    print(f"Result: {passed}/{total} passed. Error: {error}")

    # 测试用例3：超时代码
    print("\nTest 3: Timeout code")
    code3 = """
def add_two(a, b):
    while True:
        pass
    return a + b
"""

    passed, total, error = execute_code_with_tests(code3, tests1, timeout=1.0)
    print(f"Result: {passed}/{total} passed. Error: {error}")

    # 测试代码提取
    print("\nTest 4: Code extraction from markdown")
    markdown_code = """
Here's the solution:

```python
def add_two(a, b):
    return a + b
```

This function adds two numbers.
"""

    extracted = extract_code_from_completion(markdown_code)
    print(f"Extracted code:\n{extracted}")

    # 验证语法
    print("\nTest 5: Syntax validation")
    is_valid, error = validate_python_syntax(extracted)
    print(f"Syntax valid: {is_valid}, Error: {error}")

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
