#!/usr/bin/env python3
"""
ReDit GRPO Wrapper - Auto-configurable for different models
Allows running GRPO with different models without modifying grpo_gsm8k.py

Usage:
    python run_grpo.py --model_name mistralai/Mixtral-8x7B-Instruct --output_dir outputs/group_b_mixtral
    python run_grpo.py --model_name deepseek-ai/deepseek-moe-16b-base --output_dir outputs/group_b_deepseek
"""

# =================================================================================
# 配置HuggingFace缓存目录和镜像源（必须在所有导入之前）
# =================================================================================
import os
os.environ['HF_HOME'] = '/usr/storage/fwan/huggingface_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import sys
import subprocess
import re
from pathlib import Path


def detect_model_config(model_name: str) -> dict:
    """Auto-detect model configuration based on model name."""
    print(f">>> Auto-detecting configuration for model: {model_name}")

    config = {
        "model_name": model_name,
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
    }

    model_lower = model_name.lower()
    
    # 提取模型简称用于文件命名
    model_tag = model_name.split('/')[-1].lower()
    config["model_tag"] = model_tag
    
    if "mixtral" in model_lower:
        config["model_type"] = "Mixtral"
        print(">>> Mixtral-8x7B detected")
    elif "deepseek" in model_lower:
        config["model_type"] = "DeepSeek"
        print(">>> DeepSeek-MoE detected")
    elif "qwen" in model_lower:
        config["model_type"] = "Qwen"
        print(">>> Qwen detected")
    elif "olmoe" in model_lower or "olmo" in model_lower:
        config["model_type"] = "OLMoE"
        print(">>> OLMoE detected")
    elif "llama" in model_lower:
        config["model_type"] = "Llama"
        print(">>> Llama detected")
    else:
        config["model_type"] = "Generic"
        print(">>> Generic model (assuming Transformer-based)")

    return config


def get_grpo_script_for_dataset(dataset: str) -> str:
    """
    根据数据集选择对应的GRPO脚本

    Args:
        dataset: Dataset name (gsm8k, math, mbpp)

    Returns:
        GRPO script filename
    """
    script_map = {
        "gsm8k": "grpo_gsm8k.py",
        "math": "grpo_math.py",
        "mbpp": "grpo_mbpp.py"
    }

    if dataset not in script_map:
        print(f"❌ Unsupported dataset: {dataset}")
        print(f"Supported datasets: {', '.join(script_map.keys())}")
        sys.exit(1)

    return script_map[dataset]


def modify_grpo_script(
    model_name: str,
    output_dir: str,
    dataset: str = "gsm8k",
    num_gpus: int = None,
    micro_batch_size: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    max_epochs: int = None
) -> str:
    """
    Generate modified GRPO script content with the specified model, dataset, and parameters.
    Returns the modified script path.

    Args:
        model_name: Model name/path
        output_dir: Output directory for trained model
        dataset: Dataset to use (gsm8k, math, mbpp)
        num_gpus: Number of GPUs to use
        micro_batch_size: Micro batch size per GPU
        batch_size: Global batch size
        learning_rate: Learning rate
        max_epochs: Maximum epochs

    Returns:
        Path to the temporary modified script
    """
    # Get the appropriate script for the dataset
    grpo_script_name = get_grpo_script_for_dataset(dataset)
    grpo_script_path = Path(__file__).parent / grpo_script_name

    if not grpo_script_path.exists():
        print(f"❌ {grpo_script_name} not found at {grpo_script_path}")
        print(f"Please ensure {grpo_script_name} exists in {Path(__file__).parent}")
        sys.exit(1)

    print(f">>> Using GRPO script: {grpo_script_name}")

    with open(grpo_script_path, 'r') as f:
        content = f.read()

    # Replace model_name
    # Look for the pattern: model_name = "..."
    content = re.sub(
        r'model_name = "[^"]*"',
        f'model_name = "{model_name}"',
        content
    )

    # Replace output_dir if specified
    if output_dir:
        content = re.sub(
            r'output_dir = "[^"]*"',
            f'output_dir = "{output_dir}"',
            content
        )

        # Also replace run_name to match the group/dataset pattern
        # Extract model type from model_name for run_name
        model_tag = model_name.split('/')[-1].lower()
        if "mixtral" in model_name.lower():
            model_type = "mixtral"
        elif "deepseek" in model_name.lower():
            model_type = "deepseek"
        elif "qwen" in model_name.lower():
            model_type = "qwen"
        elif "olmoe" in model_name.lower() or "olmo" in model_name.lower():
            model_type = "olmoe"
        else:
            model_type = model_tag

        # Extract dataset from output_dir (last part before extension)
        dataset_name = "gsm8k"  # default
        if "math" in output_dir.lower():
            dataset_name = "math"
        elif "mbpp" in output_dir.lower():
            dataset_name = "mbpp"

        run_name = f"group_b_grpo_{model_type}_{dataset_name}"
        content = re.sub(
            r'run_name = "[^"]*"',
            f'run_name = "{run_name}"',
            content
        )

    # Replace trainable parameters if specified
    if num_gpus is not None:
        content = re.sub(
            r'num_gpus = \d+',
            f'num_gpus = {num_gpus}',
            content
        )

    if micro_batch_size is not None:
        content = re.sub(
            r'per_device_train_batch_size = \d+',
            f'per_device_train_batch_size = {micro_batch_size}',
            content
        )

    if batch_size is not None:
        content = re.sub(
            r'train_batch_size = \d+',
            f'train_batch_size = {batch_size}',
            content
        )

    if learning_rate is not None:
        content = re.sub(
            r'learning_rate = [\d.e\-]+',
            f'learning_rate = {learning_rate}',
            content
        )

    if max_epochs is not None:
        content = re.sub(
            r'num_train_epochs = \d+',
            f'num_train_epochs = {max_epochs}',
            content
        )

    # Create a temporary modified script
    temp_script = Path(__file__).parent / f"{grpo_script_name.replace('.py', '')}_temp_{Path(model_name).stem}.py"

    with open(temp_script, 'w') as f:
        f.write(content)

    print(f">>> Generated temporary script: {temp_script}")
    return str(temp_script)


def run_grpo(
    model_name: str,
    output_dir: str = None,
    dataset: str = "gsm8k",
    num_gpus: int = None,
    micro_batch_size: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    max_epochs: int = None,
    **kwargs
):
    """
    Run GRPO with specified model, dataset, and parameters using torchrun for multi-GPU training.

    Args:
        model_name: Model name or path
        output_dir: Output directory for trained model
        dataset: Dataset to use (gsm8k, math, mbpp)
        num_gpus: Number of GPUs to use (auto-detect if None)
        micro_batch_size: Micro batch size per GPU (default: 4)
        batch_size: Global batch size (default: auto)
        learning_rate: Learning rate (default: 5e-6)
        max_epochs: Maximum training epochs (default: 1)
    """
    import torch

    # Detect model config
    config = detect_model_config(model_name)

    # Generate modified script
    if output_dir is None:
        output_dir = f"outputs/group_b_grpo_{config['model_tag']}_{dataset}"

    temp_script = modify_grpo_script(
        model_name,
        output_dir,
        dataset=dataset,
        num_gpus=num_gpus,
        micro_batch_size=micro_batch_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs
    )

    # Detect number of GPUs if not specified
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("⚠️  Warning: No GPU detected, defaulting to 4")
            num_gpus = 4

    print(f"\n{'='*70}")
    print(f"🚀 GRPO Multi-GPU Training (DeepSpeed)")
    print(f"{'='*70}")
    print(f"Dataset: {dataset}")
    print(f"Model Type: {config['model_type']}")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"GPUs: {num_gpus}")
    print(f"Using temporary script: {temp_script}")
    print(f"Launch command: deepspeed --num_gpus={num_gpus} {temp_script}")
    print(f"{'='*70}\n")

    try:
        # Run with deepspeed for multi-GPU training
        result = subprocess.run(
            ["deepspeed",
             f"--num_gpus={num_gpus}",
             temp_script],
            cwd=Path(__file__).parent,
            check=False
        )

        if result.returncode == 0:
            print(f"\n{'='*70}")
            print(f"✅ GRPO training completed successfully!")
            print(f"✅ Model saved to: {output_dir}")
            print(f"✅ Multi-GPU training completed with {num_gpus} GPUs")
            print(f"{'='*70}\n")
        else:
            print(f"\n❌ GRPO training failed with return code {result.returncode}")

        return result.returncode

    except Exception as e:
        print(f"❌ Error running GRPO: {e}")
        return 1

    finally:
        # Cleanup temp script
        temp_script_path = Path(temp_script)
        if temp_script_path.exists():
            try:
                temp_script_path.unlink()
                print(f">>> Cleaned up temporary script")
            except Exception as e:
                print(f"⚠️  Could not cleanup temp script: {e}")


def main():
    """Main entry point with comprehensive parameter support."""
    parser = argparse.ArgumentParser(
        description="ReDit GRPO training wrapper with fully configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU with default settings
  python run_grpo.py --model_name Qwen/Qwen1.5-MoE-A2.7B --dataset gsm8k --num_gpus 1

  # 4 GPUs with custom batch size
  python run_grpo.py --model_name Qwen/Qwen1.5-MoE-A2.7B --dataset gsm8k --num_gpus 4

  # 8 GPUs with fast training (smaller batch size)
  python run_grpo.py --model_name Qwen/Qwen1.5-MoE-A2.7B --dataset gsm8k --num_gpus 8 --batch_size 32
        """
    )

    # 基础参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen1.5-MoE-A2.7B",
        help="Model name or path (Mixtral, DeepSeek, Qwen, OLMoE, etc.)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math", "mbpp"],
        help="Dataset to use (gsm8k, math, or mbpp)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If not specified, auto-generated based on model and dataset."
    )

    # GPU和批大小参数
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use. If not specified, auto-detect all available GPUs."
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=None,
        help="Micro batch size per GPU (default: 4). Reduce if OOM."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Global batch size (default: auto-calculated). For fast training, reduce this value."
    )

    # 训练参数
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: 5e-6)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs (default: 1 for GRPO)"
    )

    # DeepSpeed 自动添加的参数（需要接受但不使用）
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (added by DeepSpeed launcher)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ReDit GRPO Training - Configurable Parameter Wrapper")
    print("=" * 70)

    return run_grpo(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset=args.dataset,
        num_gpus=args.num_gpus,
        micro_batch_size=args.micro_batch_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs
    )


if __name__ == "__main__":
    sys.exit(main())
