#!/usr/bin/env python3
"""
HuggingFace 模型下载脚本 (与 env.sh 路径保持一致)

使用方法:
    python download_hf.py              # 按顺序下载所有预设模型
    python download_hf.py list         # 列出预设模型
    python download_hf.py model allenai/OLMoE-1B-7B-0125
    python download_hf.py model Qwen/Qwen3-30B-A3B --workers 4

下载顺序:
    1. allenai/OLMoE-1B-7B-0125
    2. allenai/OLMoE-1B-7B-0125-Instruct
    3. Qwen/Qwen1.5-MoE-A2.7B
    4. Qwen/Qwen3-30B-A3B
    5. Qwen/Qwen3-30B-A3B-Instruct-2507
    6. mistralai/Mixtral-8x7B-v0.1
    7. mistralai/Mixtral-8x7B-Instruct-v0.1
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# =============================================================================
# 自动加载 env.sh（路径统一在 scripts/env.sh 管理，此处不硬编码任何路径）
# 若环境变量已由调用方设置（如 bash 脚本 source env.sh），则不覆盖。
# =============================================================================
def _load_env_sh():
    """解析 scripts/env.sh，将其中的 export 变量注入当前进程（不覆盖已有变量）。"""
    env_sh = Path(__file__).parent.parent / "scripts" / "env.sh"
    if not env_sh.exists():
        print(f"⚠️  找不到 env.sh: {env_sh}", file=sys.stderr)
        return
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {env_sh} 2>/dev/null && env"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            key, sep, val = line.partition("=")
            if sep and key and key == key.strip() and key not in os.environ:
                os.environ[key] = val
    except Exception as e:
        print(f"⚠️  加载 env.sh 失败: {e}", file=sys.stderr)

_load_env_sh()

# env.sh 加载后，路径变量应已就绪
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", "/wutailin/hf_hub/hub"))

# =============================================================================
# 预设模型列表（按下载顺序）
# =============================================================================
PRESET_MODELS = [
    # --- OLMoE ---
    "allenai/OLMoE-1B-7B-0125",
    "allenai/OLMoE-1B-7B-0125-Instruct",
    # --- Qwen MoE ---
    "Qwen/Qwen1.5-MoE-A2.7B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    # --- DeepSeek MoE ---
    "deepseek-ai/deepseek-moe-16b-base",
    # --- Mixtral ---
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

# =============================================================================
# Retry 配置（网络不稳定时自动重连）
# =============================================================================
MAX_RETRIES     = 15          # 最大重试次数
RETRY_BASE_DELAY = 30         # 初始等待秒数
RETRY_MAX_DELAY  = 600        # 最长等待上限（10 分钟）
ETAG_TIMEOUT     = 120        # 元数据请求超时（秒）
DOWNLOAD_TIMEOUT = 600        # 单文件下载超时（秒）


# =============================================================================
# 核心下载函数
# =============================================================================

def _snapshot_with_retry(repo_id: str, repo_type: str, max_workers: int,
                          ignore_patterns: list) -> str:
    """带指数退避的 snapshot_download，自动重试直到成功或超过上限。"""
    from huggingface_hub import snapshot_download

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                resume_download=True,
                max_workers=max_workers,
                ignore_patterns=ignore_patterns,
                etag_timeout=ETAG_TIMEOUT,
            )
            return path

        except KeyboardInterrupt:
            print("\n⛔ 用户中断下载")
            raise

        except Exception as e:
            if attempt >= MAX_RETRIES:
                print(f"\n❌ 已达最大重试次数 ({MAX_RETRIES})，放弃: {repo_id}")
                raise

            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            print(f"  ⚠️  第 {attempt} 次失败: {type(e).__name__}: {e}")
            print(f"  ⏳ {delay}s 后重试 ({attempt}/{MAX_RETRIES})...")
            time.sleep(delay)


def download_model(repo_id: str, max_workers: int = 8) -> str:
    """下载单个模型到 HF_HUB_CACHE（标准 blob/snapshot 格式）。"""
    print(f"\n{'='*62}")
    print(f"  下载模型: {repo_id}")
    print(f"  缓存目录: {HF_HUB_CACHE}")
    print(f"  镜像源:   {os.environ['HF_ENDPOINT']}")
    print(f"  最大重试: {MAX_RETRIES} 次，退避上限 {RETRY_MAX_DELAY}s")
    print(f"{'='*62}")

    path = _snapshot_with_retry(
        repo_id=repo_id,
        repo_type="model",
        max_workers=max_workers,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf"],
    )
    print(f"\n✅ 完成: {repo_id}")
    print(f"   路径: {path}")
    return path


def download_dataset(repo_id: str, max_workers: int = 8) -> str:
    """下载数据集。"""
    print(f"\n{'='*62}")
    print(f"  下载数据集: {repo_id}")
    print(f"  缓存目录: {HF_HUB_CACHE}")
    print(f"{'='*62}")

    path = _snapshot_with_retry(
        repo_id=repo_id,
        repo_type="dataset",
        max_workers=max_workers,
        ignore_patterns=[],
    )
    print(f"\n✅ 完成: {repo_id}")
    print(f"   路径: {path}")
    return path


def download_all(max_workers: int = 8):
    """按顺序下载所有预设模型，单个失败不中断整体。"""
    print("\n" + "=" * 62)
    print(f"  开始批量下载，共 {len(PRESET_MODELS)} 个模型")
    print("=" * 62)

    success, failed = [], []
    total = len(PRESET_MODELS)

    for idx, repo_id in enumerate(PRESET_MODELS, 1):
        print(f"\n[{idx}/{total}] {repo_id}")
        try:
            download_model(repo_id, max_workers=max_workers)
            success.append(repo_id)
        except KeyboardInterrupt:
            print("\n⛔ 用户中断，停止批量下载")
            break
        except Exception as e:
            failed.append((repo_id, str(e)))
            print(f"  ❌ 跳过，继续下一个")

    # 结果汇总
    print("\n" + "=" * 62)
    print("  批量下载结果")
    print("=" * 62)
    print(f"\n✅ 成功 ({len(success)}/{total}):")
    for r in success:
        print(f"   - {r}")
    if failed:
        print(f"\n❌ 失败 ({len(failed)}/{total}):")
        for r, e in failed:
            print(f"   - {r}: {e}")
    print(f"\n缓存目录: {HF_HUB_CACHE}")

    return len(failed) == 0


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace 模型下载脚本（自动重试/断点续传）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=8,
        help="并行下载线程数 (默认 8)",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token（也可通过 HF_TOKEN 环境变量设置）",
    )

    subparsers = parser.add_subparsers(dest="command")

    # model
    p_model = subparsers.add_parser("model", help="下载单个模型")
    p_model.add_argument("repo_id", help="模型 repo ID，如 allenai/OLMoE-1B-7B-0125")

    # dataset
    p_ds = subparsers.add_parser("dataset", help="下载数据集")
    p_ds.add_argument("repo_id", help="数据集 repo ID")

    # list
    subparsers.add_parser("list", help="列出预设模型")

    args = parser.parse_args()

    # 设置 token
    if args.token:
        os.environ["HF_TOKEN"] = args.token

    if args.command == "model":
        download_model(args.repo_id, max_workers=args.workers)

    elif args.command == "dataset":
        download_dataset(args.repo_id, max_workers=args.workers)

    elif args.command == "list":
        print(f"预设模型列表（共 {len(PRESET_MODELS)} 个，按下载顺序）:")
        for i, m in enumerate(PRESET_MODELS, 1):
            print(f"  {i:2d}. {m}")
        print(f"\n缓存目录: {HF_HUB_CACHE}")

    else:
        # 默认：下载全部
        success = download_all(max_workers=args.workers)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
