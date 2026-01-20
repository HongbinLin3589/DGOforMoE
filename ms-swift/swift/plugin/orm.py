import os
import re
from typing import TYPE_CHECKING, Dict, List, Union

import json

if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:
    """Base class for synchronous outcome reward models (ORM).

    Subclasses should implement the __call__ method to compute rewards.

    Example:
        class MyReward(ORM):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class AsyncORM:
    """Base class for asynchronous outcome reward models (ORM).

    Use this for reward functions that involve I/O operations (e.g., API calls,
    database queries) that can benefit from async execution.

    Async reward functions are executed in parallel using asyncio.gather,
    which can significantly speed up reward computation when multiple async
    reward functions are used or when the reward function involves network calls.

    Example:
        class MyAsyncReward(AsyncORM):
            async def __call__(self, completions, **kwargs) -> List[float]:
                # Use asyncio.gather for parallel execution of all API calls
                import asyncio
                import aiohttp

                async def score_single(session, text):
                    async with session.post(api_url, json={'text': text}) as resp:
                        result = await resp.json()
                        return result['score']

                async with aiohttp.ClientSession() as session:
                    tasks = [score_single(session, c) for c in completions]
                    rewards = await asyncio.gather(*tasks)
                    return list(rewards)
    """

    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")

    def __call__(self, completions, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify

        # Extract solution from kwargs (passed by GRPO trainer via rows_to_batched)
        solution = kwargs.get('solution')
        if solution is None:
            raise ValueError("MathAccuracy requires 'solution' field in the dataset. "
                           "Make sure your dataset has a 'solution' column or use --columns to map it.")

        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class MBPPCodeExecution(ORM):
    """
    MBPP Code Execution Reward - 使用 HuggingFace Evaluate code_eval (与 lm-eval 一致)

    与 lm-evaluation-harness 完全一致的代码执行验证
    """

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('evaluate') is not None, (
            'The evaluate package is required but not installed. '
            "Please install it using 'pip install evaluate'")

        try:
            import evaluate as hf_evaluate
            self.code_eval = hf_evaluate.load("code_eval")
            self.has_code_eval = True
        except Exception as e:
            raise RuntimeError(f"Failed to load code_eval metric: {e}")

    @staticmethod
    def extract_code_blocks(text: str) -> str:
        """从文本中提取 Markdown 代码块（与 lm-eval 完全一致 + 增强回退）

        lm-eval 参考: lm-evaluation-harness/lm_eval/tasks/mbpp/utils.py
        """
        # === lm-eval 兼容逻辑（优先）===
        # 使用与 lm-eval 完全一致的模式: (?:\w+)? 匹配任何语言标记
        pattern = r"```(?:\w+)?\n?(.*?)\n?```"

        # 1. 先尝试直接匹配（处理已有```的情况）
        matches = re.findall(pattern, text, re.DOTALL)

        # 2. 如果失败，添加前缀再匹配（处理gen_prefix情况）
        if not matches:
            matches = re.findall(pattern, r"```" + text, re.DOTALL)

        # 3. 过滤空字符串并返回最后一个非空匹配
        matches = [m for m in matches if m and m.strip()]
        if matches:
            return matches[-1].strip()  # 返回最后一个非空匹配

        # === 增强回退（仅当 lm-eval 逻辑失败时）===
        # Fallback 1: 检查是否是纯代码（以 def/class 开头）
        text_stripped = text.strip()
        if text_stripped.startswith("def ") or text_stripped.startswith("class "):
            return text_stripped

        # Fallback 2: 查找文本中任何以 def/class 开头的代码
        lines = text.split('\n')
        code_started = False
        code_lines = []
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                code_started = True
            if code_started:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines).strip()

        return ""

    def __call__(self, completions, solution=None, test_list=None, **kwargs) -> List[float]:
        """
        计算 MBPP 代码执行的 reward

        Args:
            completions: 模型生成的代码文本列表（可能包含 Markdown 标记）
            solution: 测试用例列表（Python 断言语句）- 兼容旧配置
            test_list: 测试用例列表 - MBPP原始字段名

        Returns:
            rewards: 1.0 如果所有测试通过，0.0 否则

        Note:
            优先使用 test_list（MBPP原始字段），兼容 solution 参数
        """
        import sys

        # 使用 print 确保调试信息出现在 stdout（会被日志捕获）
        def debug_print(msg):
            print(f"[MBPPCodeExecution] {msg}", file=sys.stderr, flush=True)

        # 调试: 记录传入的参数
        debug_print(f"Called with {len(completions)} completions")
        debug_print(f"test_list is {'not None' if test_list is not None else 'None'} "
                   f"(type: {type(test_list).__name__ if test_list is not None else 'N/A'}, "
                   f"len={len(test_list) if test_list is not None else 0})")
        debug_print(f"solution is {'not None' if solution is not None else 'None'} "
                   f"(type: {type(solution).__name__ if solution is not None else 'N/A'})")
        debug_print(f"Other kwargs keys: {list(kwargs.keys())}")

        # 优先使用 test_list（避免与MBPP的solution字段冲突）
        tests = test_list if test_list is not None else solution
        if tests is None:
            debug_print("ERROR: Neither test_list nor solution provided!")
            raise ValueError("MBPPCodeExecution requires either 'test_list' or 'solution' parameter")

        rewards = []
        num_extracted = 0
        num_passed = 0

        # tests应该是所有completion共享的测试用例列表，不是每个completion一个
        # 如果tests是嵌套列表（每个样本有不同的测试），保持原逻辑
        # 如果tests是简单列表（所有样本共享测试），broadcast到所有completions
        if tests and isinstance(tests[0], list):
            # tests是嵌套列表：[[test1, test2], [test3, test4], ...]
            # 这是batch处理模式，每个completion有不同的测试
            test_cases_list = tests
        else:
            # tests是简单列表：[test1, test2, ...]
            # 这是单样本模式，所有completions共享相同的测试
            test_cases_list = [tests] * len(completions)

        for idx, (completion, test_cases) in enumerate(zip(completions, test_cases_list)):
            # 1. 提取代码块
            code = self.extract_code_blocks(completion)

            # 如果没有提取到代码，返回 0
            if not code or code.strip() == "":
                if idx == 0:  # 只记录第一个样本的详情
                    debug_print(f"Sample 0: No code extracted from completion: {completion[:300]}...")
                rewards.append(0.0)
                continue

            num_extracted += 1

            try:
                # 2. 组合测试用例为一个测试字符串
                # code_eval 期望 references 是字符串列表，每个字符串包含所有要执行的测试代码
                # 格式: ["assert1\nassert2\nassert3"] 而不是 [["assert1", "assert2", "assert3"]]
                if isinstance(test_cases, list):
                    combined_tests = "\n".join(test_cases)
                else:
                    combined_tests = test_cases

                if idx == 0:  # 记录第一个样本的详情
                    debug_print(f"Sample 0 code (first 200 chars): {code[:200]}...")
                    debug_print(f"Sample 0 tests: {combined_tests[:200]}...")

                # 3. 使用 code_eval 执行代码并验证测试用例
                # references: 合并后的测试字符串列表
                # predictions: 代码候选列表（嵌套列表格式）
                results, logs = self.code_eval.compute(
                    references=[combined_tests],  # 合并后的测试字符串
                    predictions=[[code]],  # 生成的代码
                    k=[1]  # Pass@1
                )

                # 4. 返回 Pass@1 结果（0.0 或 1.0）
                reward = float(results["pass@1"])
                rewards.append(reward)
                if reward > 0:
                    num_passed += 1

                if idx == 0:
                    debug_print(f"Sample 0 result: pass@1={reward}")

            except Exception as e:
                # 代码执行失败（语法错误、运行时错误等）
                if idx == 0:
                    debug_print(f"Sample 0 error: {e}")
                    debug_print(f"Sample 0 code was: {code[:300]}...")
                rewards.append(0.0)

        debug_print(f"Summary: {num_extracted}/{len(completions)} code extracted, "
                   f"{num_passed}/{len(completions)} passed")

        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, **kwargs) -> List[float]:
        # Extract solution from kwargs (will be passed to accuracy_orm)
        solution = kwargs.get('solution')
        if solution is None:
            raise ValueError("CosineReward requires 'solution' field in the dataset.")

        # Pass solution through kwargs to the accuracy ORM
        acc_rewards = self.accuracy_orm(completions, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'mbpp': MBPPCodeExecution,  # MBPP 代码执行验证 (与 lm-eval 一致)
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}
