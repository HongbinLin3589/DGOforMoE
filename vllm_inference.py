# =================================================================================
# 配置HuggingFace缓存目录和镜像源（必须在所有导入之前）
# =================================================================================
import os
os.environ['HF_HOME'] = '/usr/storage/fwan/huggingface_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import json
import re
import sys
from pathlib import Path
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =================================================================================
# 导入统一的数据集配置
# =================================================================================

# 添加 DGO 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))
from datasets_config import DatasetLoader, SYSTEM_PROMPTS, EXTRACT_ANSWER_FUNCS

# =================================================================================
# 数据集特定的 prompt 格式
# =================================================================================

SYSTEM_PROMPTS_CUSTOM = {
    "gsm8k": "You are an expert in math reasoning. Your goal is to help the user solve math problems. You should think step by step and give the final answer in the format <answer>ANSWER</answer>.",
    "math": "You are an expert in mathematical problem solving. Solve the given problem step by step and provide the final answer in a box using \\boxed{answer} format.",
    "mbpp": "You are an expert Python programmer. Write clean, efficient, and correct Python code to solve the given problem."
}

XML_COT_FORMAT = "<thinking>\n{reasoning}\n</thinking>\n<answer>\n{answer}\n</answer>"

# Few-shot examples for each dataset
FEW_SHOT_EXAMPLES = {
    "gsm8k": {
        'question': 'What is the largest single-digit prime number?',
        'assistant': XML_COT_FORMAT.format(
            reasoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
            answer="7"
        )
    },
    "math": {
        'question': 'What is the value of 2 + 2?',
        'assistant': 'The answer is $2 + 2 = 4$. Therefore, the answer is \\boxed{4}.'
    },
    "mbpp": {
        'question': 'Write a function to return the sum of two numbers.',
        'assistant': 'def sum_two(a, b):\n    """Return the sum of two numbers."""\n    return a + b'
    }
}

def extract_hash_answer(text: str) -> str | None:
    """从文本中提取 #### XXXX 格式的答案。"""
    match = re.search(r"####\s*([\d,]+)", text)
    if match:
        return match.group(1).replace(',', '')
    return None

def extract_math_answer(text: str) -> str | None:
    """从 MATH 格式提取答案 (\\boxed{...})"""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None

def load_dataset_by_type(dataset_name: str, split: str = "train") -> Dataset:
    """
    统一加载不同数据集。

    Args:
        dataset_name: 数据集名称 (gsm8k, math, mbpp)
        split: 数据集分割 (train, test)

    Returns:
        加载的数据集
    """
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower not in ["gsm8k", "math", "mbpp"]:
        raise ValueError(f"不支持的数据集: {dataset_name_lower}. 支持: gsm8k, math, mbpp")

    print(f">>> 加载数据集: {dataset_name_lower} (split={split})")

    try:
        if dataset_name_lower == "gsm8k":
            # GSM8K 从本地 parquet 文件加载
            file_path = f'/usr/commondata/public/hf_hub/cc/DGO/ReDit/dataset/gsm8k/main/{split}-00000-of-00001.parquet'
            dataset = load_dataset('parquet', data_files=file_path)
            actual_split = 'train' if 'train' in dataset else list(dataset.keys())[0]
            data = dataset[actual_split]

        elif dataset_name_lower == "math":
            # MATH 从 HuggingFace 加载
            dataset = load_dataset('hendrycks/math', 'all')
            if split not in dataset:
                print(f"警告: MATH 数据集中不存在 split '{split}', 使用 'train'")
                split = 'train'
            data = dataset[split]

        elif dataset_name_lower == "mbpp":
            # MBPP 从 HuggingFace 加载
            dataset = load_dataset('mbpp')
            if split not in dataset:
                print(f"警告: MBPP 数据集中不存在 split '{split}', 使用 'train'")
                split = 'train'
            data = dataset[split]

    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        raise

    print(f">>> 数据集加载成功，包含 {len(data)} 条数据")
    return data

def preprocess_dataset(dataset_name: str, data: Dataset, system_prompt: str = None) -> Dataset:
    """
    预处理数据集为推理格式。

    Args:
        dataset_name: 数据集名称
        data: 原始数据集
        system_prompt: 系统提示词 (可选)

    Returns:
        处理后的数据集
    """
    dataset_name_lower = dataset_name.lower()

    if system_prompt is None:
        system_prompt = SYSTEM_PROMPTS_CUSTOM.get(dataset_name_lower, "You are a helpful assistant.")

    print(f">>> 预处理数据集...")

    if dataset_name_lower == "gsm8k":
        # GSM8K 格式处理
        def preprocess_gsm8k(x):
            example = FEW_SHOT_EXAMPLES["gsm8k"]
            return {
                'prompt': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': example['question']},
                    {'role': 'assistant', 'content': example['assistant']},
                    {'role': 'user', 'content': x['question']}
                ],
                'answer': extract_hash_answer(x['answer'])
            }
        data = data.map(preprocess_gsm8k)

    elif dataset_name_lower == "math":
        # MATH 格式处理
        def preprocess_math(x):
            example = FEW_SHOT_EXAMPLES["math"]
            return {
                'prompt': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': example['question']},
                    {'role': 'assistant', 'content': example['assistant']},
                    {'role': 'user', 'content': x['problem']}
                ],
                'answer': extract_math_answer(x['solution'])
            }
        data = data.map(preprocess_math)

    elif dataset_name_lower == "mbpp":
        # MBPP 格式处理
        def preprocess_mbpp(x):
            example = FEW_SHOT_EXAMPLES["mbpp"]
            return {
                'prompt': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': example['question']},
                    {'role': 'assistant', 'content': example['assistant']},
                    {'role': 'user', 'content': x['text']}
                ],
                'answer': x['test_list']
            }
        data = data.map(preprocess_mbpp)

    print(f">>> 预处理完成")
    return data

# =================================================================================
# 主推理逻辑
# =================================================================================

def main(args):
    """
    使用 vLLM 对指定数据集进行快速推理。
    """
    # 加载模型
    print(f"正在加载模型: {args.model_name}")
    print(f">>> vLLM配置:")
    print(f"    tensor_parallel_size: {args.tensor_parallel_size}")
    print(f"    gpu_memory_utilization: {args.gpu_memory_utilization}")
    print(f"    max_num_seqs: {args.max_num_seqs}")
    print(f"    max_num_batched_tokens: {args.max_num_batched_tokens}")

    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # 批量推理优化参数
        max_num_seqs=args.max_num_seqs,  # 并行处理的最大序列数
        max_num_batched_tokens=args.max_num_batched_tokens,  # 每批最大token数
        swap_space=args.swap_space,  # CPU swap空间(GB)，用于更大batch
        dtype="bfloat16",  # 使用bf16节省显存
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 为没有 chat_template 的 base model 设置默认模板
    # 使用 ms-swift 的 "default" 模板格式，保持和 GRPO 一致
    if tokenizer.chat_template is None:
        print(">>> 模型没有chat_template，设置ms-swift default模板...")
        tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}

{% set messages = messages[1:] %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}### Human:
{{ message['content'] }}

{% elif message['role'] == 'assistant' %}### Assistant:
{{ message['content'] }}

{% endif %}{% endfor %}{% if add_generation_prompt %}### Assistant:
{% endif %}"""

    # 加载并预处理数据集
    print(f"正在加载数据集: {args.dataset}, split={args.dataset_split}")
    try:
        dataset = load_dataset_by_type(args.dataset, split=args.dataset_split)
        dataset = preprocess_dataset(args.dataset, dataset)
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return

    # 准备 prompts
    # 将聊天记录转换为单个字符串
    print(f">>> 准备推理 prompts...")
    prompts = [tokenizer.apply_chat_template(item['prompt'], tokenize=False, add_generation_prompt=True) for item in dataset]
    ground_truth_answers = [item['answer'] for item in dataset]

    # 配置采样参数
    print(f">>> Stop tokens: {args.stop}")  # 调试输出

    # 检查是否使用了 </answer> 作为 stop token
    # vLLM 默认会从输出中移除 stop token，需要设置 include_stop_str_in_output=True
    include_stop_in_output = args.stop is not None and '</answer>' in args.stop

    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
        include_stop_str_in_output=include_stop_in_output  # 保留 stop token 在输出中
    )

    # 运行推理
    print("开始推理...")
    outputs = llm.generate(prompts, sampling_params)
    print("推理完成。")

    # 处理并保存结果
    results = []
    fixed_count = 0
    total_count = 0

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_texts = []

        for o in output.outputs:
            text = o.text
            total_count += 1

            # 后处理：确保 <answer> 标签正确闭合
            # 如果有 <answer> 但没有 </answer>，添加闭合标签
            if '<answer>' in text and '</answer>' not in text:
                # 找到最后一个 <answer> 的位置，在文本末尾添加 </answer>
                text = text.rstrip() + '\n</answer>'
                fixed_count += 1

            generated_texts.append(text)

        results.append({
            "prompt": prompt,
            "generated_texts": generated_texts,
            "ground_truth_answer": ground_truth_answers[i]
        })

    if fixed_count > 0:
        print(f">>> 后处理: 修复了 {fixed_count}/{total_count} 个缺少 </answer> 闭合标签的响应")

    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"✅ 结果已保存至 {args.output_file}")
    except IOError as e:
        print(f"❌ 无法写入输出文件 '{args.output_file}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 对多个数据集进行快速推理（支持 GSM8K, MATH, MBPP）")
    parser.add_argument("--model_name", type=str, default="mistralai/Mixtral-8x7B-Instruct", help="用于推理的模型名称或路径。")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math", "mbpp"], help="要推理的数据集 (gsm8k, math, mbpp)。")
    parser.add_argument("--dataset_split", type=str, default="train", help="要使用的数据集 split ('train' 或 'test')。")
    parser.add_argument("--output_file", type=str, default=None, help="保存推理结果的文件。如不指定，自动生成为 inference_results_{dataset}.json。")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="使用的 GPU 数量。")
    parser.add_argument("--n", type=int, default=8, help="为每个 prompt 生成的输出序列数量。")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度。")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p 采样。")
    parser.add_argument("--max_tokens", type=int, default=1024, help="生成的最大 token 数量。")
    parser.add_argument("--stop", type=str, action='append', default=None,
                        help="停止序列, 可多次指定。例如: --stop '</answer>' --stop '### Human:'")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU 内存使用率 (0.0-1.0)。")

    # 批量推理优化参数
    parser.add_argument("--max_num_seqs", type=int, default=256,
                        help="并行处理的最大序列数。增大可提高吞吐量，但需要更多显存。默认256。")
    parser.add_argument("--max_num_batched_tokens", type=int, default=None,
                        help="每批最大token数。None=自动。可设为8192/16384/32768等。")
    parser.add_argument("--swap_space", type=float, default=4,
                        help="CPU swap空间(GB)，用于支持更大batch。默认4GB。")

    args = parser.parse_args()

    # 如果未指定输出文件，自动生成（包含模型名称以避免覆盖）
    if args.output_file is None:
        # 提取模型简称（例如 mistralai/Mixtral-8x7B-v0.1 -> mixtral-8x7b-v0.1）
        model_tag = args.model_name.split('/')[-1].lower()
        args.output_file = f"inference_results_{model_tag}_{args.dataset}.json"

    main(args)