#!/usr/bin/env python3
"""
数据集配置和加载器 - 支持GSM8K, MATH, MBPP

统一的数据集接口，自动处理不同数据集的差异
"""

from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset, concatenate_datasets
import re


# ============================================================================
# 数据集配置
# ============================================================================

DATASET_CONFIGS = {
    "gsm8k": {
        "name": "GSM8K",
        "description": "Grade School Math 8K - 数学问题求解 (与 lm-eval 一致)",
        "hf_name": "gsm8k",                      # lm-eval 标准：简写形式
        "hf_config": "main",
        "question_field": "question",
        "answer_field": "answer",
        "extract_answer_func": "extract_gsm8k_answer",
        "max_length": 1024,
        "supported_splits": ["train", "test"],
    },
    "math": {
        "name": "MATH",
        "description": "MATH - 竞技数学问题 (5000题高中/大学数学题，与 lm-eval 一致)",
        "hf_name": "EleutherAI/hendrycks_math",  # lm-eval 标准
        "hf_config": None,  # 将自动加载所有子集并合并
        "hf_subsets": [  # MATH 有7个子集
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus"
        ],
        "question_field": "problem",
        "answer_field": "solution",
        "extract_answer_func": "extract_math_answer",
        "max_length": 2048,
        "supported_splits": ["train", "test"],
    },
    "mbpp": {
        "name": "MBPP",
        "description": "MBPP - Mostly Basic Python Problems (full子集: train=374, test=500，与 lm-eval 一致)",
        "hf_name": "google-research-datasets/mbpp",  # lm-eval 标准
        "hf_config": "full",                         # lm-eval 使用 full 子集（train: 374条）
        "question_field": "text",                    # 问题描述 (原始数据集)
        "answer_field": "code",                      # 参考代码 (原始数据集)
        "test_field": "test_list",                   # 测试用例列表（新增）
        "extract_answer_func": "extract_mbpp_answer",
        "max_length": 2048,
        "supported_splits": ["train", "test", "validation", "prompt"],
        # NOTE: MS-SWIFT training requires preprocessed dataset at:
        # /usr/commondata/public/hf_hub/cc/DGO/datasets/mbpp_preprocessed
        # Columns renamed: text->problem, code->solution (to match MS-SWIFT's ResponsePreprocessor)
    },
}

SYSTEM_PROMPTS = {
    "gsm8k": "You are an expert in math reasoning. Your goal is to help the user solve math problems. You should think step by step and give the final answer in the format ####ANSWER.",
    "math": "You are an expert in mathematical problem solving. Solve the given problem step by step and provide the final answer clearly.",
    "mbpp": "You are an expert Python programmer. Write clean, efficient, and correct Python code to solve the given problem.",
}


# ============================================================================
# 答案提取函数
# ============================================================================

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """从GSM8K格式提取答案: #### ANSWER"""
    match = re.search(r"####\s*([\d,\.]+)", text)
    if match:
        return match.group(1).replace(',', '')
    return None


def extract_math_answer(text: str) -> Optional[str]:
    """从MATH格式提取答案"""
    # MATH数据集中答案通常在最后
    lines = text.strip().split('\n')
    if lines:
        return lines[-1].strip()
    return None


def extract_mbpp_answer(text: str) -> Optional[str]:
    """从MBPP提取答案 - 直接返回代码"""
    return text.strip() if text else None


EXTRACT_ANSWER_FUNCS = {
    "gsm8k": extract_gsm8k_answer,
    "math": extract_math_answer,
    "mbpp": extract_mbpp_answer,
}


# ============================================================================
# 数据集加载器
# ============================================================================

class DatasetLoader:
    """统一的数据集加载接口"""

    @staticmethod
    def get_config(dataset_name: str) -> Dict:
        """获取数据集配置"""
        dataset_name = dataset_name.lower()
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(
                f"不支持的数据集: {dataset_name}\n"
                f"支持的数据集: {', '.join(DATASET_CONFIGS.keys())}"
            )
        return DATASET_CONFIGS[dataset_name]

    @staticmethod
    def list_supported_datasets() -> List[str]:
        """列出支持的所有数据集"""
        return list(DATASET_CONFIGS.keys())

    @staticmethod
    def get_dataset_info(dataset_name: str) -> str:
        """获取数据集信息"""
        config = DatasetLoader.get_config(dataset_name)
        return f"{config['name']}: {config['description']}"

    @staticmethod
    def load_dataset(
        dataset_name: str,
        split: str = "train",
        limit: Optional[int] = None
    ) -> Dataset:
        """
        加载数据集，统一处理不同数据集的差异

        Args:
            dataset_name: 数据集名称 (gsm8k, math, mbpp)
            split: 数据集分割 (train, test)
            limit: 限制数据集大小 (可选)

        Returns:
            加载的数据集
        """
        config = DatasetLoader.get_config(dataset_name)
        dataset_name_lower = dataset_name.lower()

        print(f">>> 加载数据集: {config['name']}")
        print(f">>> Split: {split}")

        # 从HuggingFace加载
        hf_subsets = config.get('hf_subsets')

        if hf_subsets:
            # MATH等数据集需要加载多个子集并合并
            print(f">>> 加载 {len(hf_subsets)} 个子集: {', '.join(hf_subsets)}")
            subset_datasets = []
            for subset in hf_subsets:
                print(f"    - 加载子集: {subset}")
                subset_ds = load_dataset(config['hf_name'], subset, split=split)
                subset_datasets.append(subset_ds)

            # 合并所有子集
            data = concatenate_datasets(subset_datasets)
            print(f">>> 合并完成，共 {len(data)} 条数据")
        elif config['hf_config']:
            dataset = load_dataset(config['hf_name'], config['hf_config'])
            # 获取指定split
            if split in dataset:
                data = dataset[split]
            else:
                available_splits = list(dataset.keys())
                raise ValueError(
                    f"数据集 {dataset_name} 中不存在 split '{split}'\n"
                    f"可用的 splits: {available_splits}"
                )
        else:
            dataset = load_dataset(config['hf_name'])
            # 获取指定split
            if split in dataset:
                data = dataset[split]
            else:
                available_splits = list(dataset.keys())
                raise ValueError(
                    f"数据集 {dataset_name} 中不存在 split '{split}'\n"
                    f"可用的 splits: {available_splits}"
                )

        # 限制数据集大小
        if limit and limit > 0:
            data = data.select(range(min(limit, len(data))))
            print(f">>> 限制数据集大小: {limit}")

        print(f">>> 数据集大小: {len(data)}")
        return data

    @staticmethod
    def preprocess_dataset(
        dataset_name: str,
        data: Dataset,
        system_prompt: Optional[str] = None
    ) -> Dataset:
        """
        预处理数据集，统一格式

        Args:
            dataset_name: 数据集名称
            data: 原始数据集
            system_prompt: 系统提示词 (可选，使用默认值)

        Returns:
            预处理后的数据集
        """
        config = DatasetLoader.get_config(dataset_name)
        dataset_name_lower = dataset_name.lower()

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPTS.get(
                dataset_name_lower,
                "You are a helpful assistant."
            )

        question_field = config['question_field']
        answer_field = config['answer_field']
        extract_func = EXTRACT_ANSWER_FUNCS[dataset_name_lower]

        print(f">>> 预处理数据集...")

        def preprocess_fn(x):
            question = x[question_field]
            answer = x[answer_field]

            return {
                'question': question,
                'answer': answer,
                'extracted_answer': extract_func(answer) if answer else None,
                'system_prompt': system_prompt,
            }

        processed = data.map(preprocess_fn, remove_columns=data.column_names)
        print(f">>> 预处理完成")

        return processed


# ============================================================================
# 快速接口
# ============================================================================

def get_dataset(
    dataset_name: str,
    split: str = "train",
    limit: Optional[int] = None,
    preprocess: bool = True,
    system_prompt: Optional[str] = None
) -> Dataset:
    """
    快速加载和预处理数据集

    Usage:
        dataset = get_dataset("gsm8k", split="train", limit=1000)
        dataset = get_dataset("math", split="test")
        dataset = get_dataset("mbpp", split="train", preprocess=False)
    """
    data = DatasetLoader.load_dataset(dataset_name, split=split, limit=limit)

    if preprocess:
        data = DatasetLoader.preprocess_dataset(
            dataset_name,
            data,
            system_prompt=system_prompt
        )

    return data


def list_datasets() -> None:
    """打印所有支持的数据集"""
    print("\n" + "="*70)
    print("支持的数据集列表:")
    print("="*70)
    for dataset_name in DatasetLoader.list_supported_datasets():
        info = DatasetLoader.get_dataset_info(dataset_name)
        print(f"  • {info}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # 测试
    list_datasets()

    # 加载示例
    print("\n示例1: 加载GSM8K")
    gsm8k = get_dataset("gsm8k", split="train", limit=5)
    print(f"加载了 {len(gsm8k)} 条数据")

    print("\n示例2: 加载MATH")
    math = get_dataset("math", split="train", limit=5)
    print(f"加载了 {len(math)} 条数据")

    print("\n示例3: 加载MBPP")
    mbpp = get_dataset("mbpp", split="train", limit=5)
    print(f"加载了 {len(mbpp)} 条数据")
