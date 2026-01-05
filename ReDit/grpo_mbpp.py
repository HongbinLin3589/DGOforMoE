#!/usr/bin/env python3
"""
GRPO Training for MBPP Dataset
基于grpo_gsm8k.py修改，使用lm-evaluation-harness的代码执行

This script implements Group B (GRPO) training specifically for MBPP dataset.
Uses safe code execution via mbpp_utils.py which wraps lm-evaluation-harness.

Plan A Implementation: Lightweight adapter leveraging lm-evaluation-harness
instead of reimplementing code execution from scratch.
"""

# =============================================================================
# 配置HuggingFace缓存目录和镜像源（必须在所有导入之前）
# =============================================================================
import os
os.environ['HF_HOME'] = '/usr/storage/fwan/huggingface_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Skip CUDA extensions compilation to avoid CUDA version mismatch issues
# (System CUDA 11.5 vs PyTorch compiled with CUDA 12.1)
os.environ['DEEPSPEED_SKIP_CUDA_EXTENSIONS'] = '1'

import math
import re
import torch
import random
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from typing import List, Callable
import sys

# 导入MBPP执行工具（方案A：lm-eval-harness集成）
try:
    from mbpp_utils import (
        extract_code_from_completion,
        execute_code_with_tests,
        validate_python_syntax
    )
except ImportError:
    print("Error: mbpp_utils.py not found. Please ensure mbpp_utils.py is in the same directory.")
    sys.exit(1)

# ========== 多卡训练：GPU 检测和梯度累积动态计算 ==========
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    print("⚠️  Warning: No GPU detected, defaulting to 8")
    num_gpus = 8

# 梯度累积动态计算（保持全局批大小 = 256）
# MBPP 使用 per_device_train_batch_size = 4（因为代码执行耗时）
global_batch_size = 256
per_device_train_batch_size = 4
gradient_accumulation_steps = global_batch_size // (num_gpus * per_device_train_batch_size)

print(f">>> GPUs detected: {num_gpus}")
print(f">>> Per-device batch size: {per_device_train_batch_size}")
print(f">>> Gradient accumulation steps: {gradient_accumulation_steps}")
print(f">>> Global batch size: {num_gpus * per_device_train_batch_size * gradient_accumulation_steps}")

# =============================================================================
# 系统提示和格式
# =============================================================================

SYSTEM_PROMPT = """You are an expert Python programmer.
Write clean, efficient, and correct Python code to solve the given problem.
Include proper function definitions with docstrings.
Make sure to follow the exact function signature provided in the problem."""

FEW_SHOT_EXAMPLE = {
    'text': 'Write a function to return the sum of two numbers.',
    'code': 'def sum_two(a, b):\n    """Return the sum of two numbers."""\n    return a + b',
    'test_list': [
        'assert sum_two(3, 4) == 7',
        'assert sum_two(0, 0) == 0',
        'assert sum_two(-1, 1) == 0'
    ]
}


# =============================================================================
# 数据集加载和预处理
# =============================================================================

def get_mbpp_dataset(split='train', limit: int = None) -> Dataset:
    """
    加载MBPP数据集

    Args:
        split: 'train' or 'test'
        limit: 最大样本数（用于快速测试）

    Returns:
        处理后的数据集
    """
    print(f"Loading MBPP dataset ({split} split)...")
    data = load_dataset('mbpp')[split]

    # 限制样本数用于快速测试
    if limit:
        data = data.select(range(min(limit, len(data))))
        print(f"Limited to {len(data)} samples")

    def preprocess(x):
        # MBPP字段: text (问题), code (参考代码), test_list (测试用例)
        return {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': FEW_SHOT_EXAMPLE['text']},
                {'role': 'assistant', 'content': FEW_SHOT_EXAMPLE['code']},
                {'role': 'user', 'content': x['text']}
            ],
            'reference_code': x['code'],  # 参考代码（不用于训练，仅供参考）
            'test_cases': x['test_list']  # 测试用例列表
        }

    return data.map(preprocess)


# =============================================================================
# 奖励函数（代码执行）
# =============================================================================

def correctness_reward_func_with_noise(
    prompts: List[str],
    completions: List[str],
    answer: List[List[str]],  # answer = test_cases
    **kwargs
) -> List[float]:
    """
    核心奖励函数：执行代码并返回pass/fail（二元奖励）

    奖励策略（对齐GSM8K）：
    - 全部测试通过: 2.0（对齐GSM8K的correctness_reward权重）
    - 任何失败: 0.0（不给部分分，鼓励完全正确）

    Args:
        prompts: 输入提示
        completions: 模型生成的代码
        answer: 每个提示对应的测试用例列表

    Returns:
        每个completion的奖励值列表（带Gaussian噪声）
    """
    original_rewards = []

    for completion, test_cases in zip(completions, answer):
        try:
            # 提取代码
            code = extract_code_from_completion(completion)

            if not code or code.strip() == '':
                original_rewards.append(0.0)
                continue

            # 快速语法检查
            is_valid, error = validate_python_syntax(code)
            if not is_valid:
                original_rewards.append(0.0)
                continue

            # 执行测试（带2秒超时保护）
            passed, total, exec_error = execute_code_with_tests(code, test_cases, timeout=2.0)

            # 二元奖励：全部通过给2.0，否则0.0（与GSM8K对齐）
            if passed == total and not exec_error:
                original_rewards.append(2.0)
            else:
                original_rewards.append(0.0)

        except Exception as e:
            # 未预期的异常
            print(f"Warning: Unexpected error in correctness_reward_func: {e}")
            original_rewards.append(0.0)

    # 添加高斯噪声（与MATH保持一致）
    noisy_rewards = [r + random.gauss(0, 0.1) for r in original_rewards]
    return noisy_rewards


def syntax_reward_func_with_noise(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    语法正确性奖励（辅助，0.5满分）

    奖励策略（二元）：
    - Python语法正确: 0.5
    - 语法错误: 0.0

    返回值带Gaussian噪声
    """
    original_rewards = []

    for completion in completions:
        try:
            code = extract_code_from_completion(completion)
            is_valid, _ = validate_python_syntax(code)
            original_rewards.append(0.5 if is_valid else 0.0)
        except Exception:
            original_rewards.append(0.0)

    # 添加高斯噪声
    noisy_rewards = [r + random.gauss(0, 0.05) for r in original_rewards]
    return noisy_rewards


def format_reward_func_with_noise(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    代码格式奖励（辅助，0.5满分）

    检查代码结构完整性：
    - 有def定义: +0.125
    - 有docstring: +0.125
    - 有return语句: +0.125
    - 正确的缩进: +0.125

    总计0.5分。返回值带Gaussian噪声
    """
    original_rewards = []

    for completion in completions:
        try:
            code = extract_code_from_completion(completion)

            score = 0.0

            # 检查代码特征（各占0.125）
            if 'def ' in code:
                score += 0.125
            if '"""' in code or "'''" in code:
                score += 0.125
            if 'return ' in code:
                score += 0.125

            # 检查缩进（简单检查）
            lines = code.split('\n')
            has_proper_indent = any(line.startswith('    ') for line in lines if line.strip())
            if has_proper_indent:
                score += 0.125

            original_rewards.append(min(score, 0.5))  # 最多0.5

        except Exception:
            original_rewards.append(0.0)

    # 添加高斯噪声
    noisy_rewards = [r + random.gauss(0, 0.05) for r in original_rewards]
    return noisy_rewards


# =============================================================================
# 注：噪声已直接集成到各奖励函数中（与MATH设计一致）
# =============================================================================


# =============================================================================
# 模型架构检测和配置
# =============================================================================

def get_lora_target_modules(model_name: str) -> List[str]:
    """
    根据模型架构自动选择 LoRA 目标模块
    仅对注意力模块应用 LoRA，不对 MoE 路由器应用
    对路由器应用 LoRA 会破坏负载均衡机制和 auxiliary loss 计算

    Args:
        model_name: 模型名称

    Returns:
        LoRA 目标模块列表
    """
    # 所有模型：只对注意力投影层应用 LoRA
    # 不包括 'gate' (MoE 路由器) 以保护负载均衡机制
    return ["q_proj", "k_proj"]


# =============================================================================
# 主训练配置
# =============================================================================

def main():
    """主训练函数"""

    # 模型配置（需要根据实际模型修改）
    model_name = "mistralai/Mixtral-8x7B-Instruct"
    output_dir = "outputs/group_b_grpo_mixtral_mbpp"
    run_name = "group_b_grpo_mixtral_mbpp"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GRPO Training Configuration for MBPP Dataset")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Run Name: {run_name}")
    print()

    # ========== 数据集 ==========
    print("Loading training dataset...")
    train_dataset = get_mbpp_dataset('train', limit=None)
    print(f"Dataset size: {len(train_dataset)} samples")
    print()

    # ========== 模型和Tokenizer ==========
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    print(f"Model loaded: {model.dtype}")
    print()

    # ========== LoRA 配置 ==========
    print("Configuring LoRA...")
    target_modules = get_lora_target_modules(model_name)
    print(f">>> LoRA target modules: {target_modules}")
    print(f">>> Supports all 4 models: Mixtral, DeepSeek, Qwen, OLMoE")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none"
    )
    print()

    # ========== GRPO训练配置 ==========
    print("Configuring GRPO training...")
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        max_steps=1000,
        num_generations=8,  # 每个prompt生成8个候选（与DGO保持一致）
        max_prompt_length=512,
        max_completion_length=512,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        report_to="wandb",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        logging_dir=f"{output_dir}/logs"
    )
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Num generations: {training_args.num_generations}")
    print(f"Learning rate: {training_args.learning_rate}")
    print()

    # ========== 奖励函数设置 ==========
    # 注：所有奖励函数已内置Gaussian噪声（ReDit风格）
    print("Setting up reward functions (noise already included)...")
    print("- correctness_reward_with_noise (2.0 max): Code execution results")
    print("- syntax_reward_with_noise (0.5 max): Python syntax validation")
    print("- format_reward_with_noise (0.5 max): Code structure incentives")
    print("Total max reward: 3.0 points")
    print()

    reward_funcs = [
        correctness_reward_func_with_noise,  # 主要奖励：2.0分
        syntax_reward_func_with_noise,       # 辅助奖励：0.5分
        format_reward_func_with_noise        # 辅助奖励：0.5分
    ]

    # ========== 创建Trainer ==========
    print("Creating GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        reward_funcs=reward_funcs
    )
    print()

    # ========== 开始训练 ==========
    print("=" * 70)
    print("Starting GRPO training on MBPP dataset")
    print("=" * 70)
    print()

    trainer.train()

    # ========== 保存最终模型 ==========
    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)

    final_model_path = f"{output_dir}/final_model"
    trainer.save_model(final_model_path)
    print(f"✅ Model saved to: {final_model_path}")
    print()

    # 保存训练配置信息
    config_info = f"""
GRPO Training Completion Report
================================
Dataset: MBPP (Python code generation)
Model: {model_name}
Output Directory: {output_dir}
Final Model: {final_model_path}

Training Configuration:
- Batch size: {training_args.per_device_train_batch_size}
- Learning rate: {training_args.learning_rate}
- Max Steps: {training_args.max_steps}
- Generations per prompt: {training_args.num_generations}
- LoRA rank: 8
- LoRA alpha: 64

Reward Functions (Total: 3.0 points max):
1. Correctness Reward (primary): Code execution against test cases
   - Full pass (all tests): 2.0
   - Any failure: 0.0
   - Gaussian noise: σ=0.1

2. Syntax Reward (auxiliary): Python syntax validation
   - Valid syntax: 0.5
   - Invalid syntax: 0.0
   - Gaussian noise: σ=0.05

3. Format Reward (auxiliary): Code structure completeness
   - Has def: +0.125
   - Has docstring: +0.125
   - Has return: +0.125
   - Has proper indentation: +0.125
   - Max: 0.5
   - Gaussian noise: σ=0.05

Design Pattern: Aligned with GSM8K and MATH (binary rewards + Gaussian noise)

Expected Performance:
- MBPP pass@1: 35-45% (Group B, comparable to ReDit GRPO on other datasets)
- Group hierarchy: Group D (DGO) > Group C (DGO trainable) > Group B (GRPO) > Group A (SFT)
"""

    with open(f"{output_dir}/training_info.txt", 'w') as f:
        f.write(config_info)

    print("✅ Training info saved!")


if __name__ == "__main__":
    main()
