#!/bin/bash
# 串行运行实验 - 简单版
# 用法: ./run_eval_serial.sh

cd /usr/commondata/public/hf_hub/cc/DGO/scripts

# Qwen
# bash ./run_eval.sh /usr/commondata/public/hf_hub/cc/DGO/outputs/swift_sft/qwen_gsm8k/v2-20251228-233904/checkpoint-1000 gsm8k_cot 64
# bash ./run_eval.sh /usr/commondata/public/hf_hub/cc/DGO/outputs/swift_dgo/qwen_gsm8k/v0-20260105-103318/checkpoint-1000 gsm8k_cot 64

# DeepSeek
# bash ./run_eval.sh /usr/commondata/public/hf_hub/cc/DGO/outputs/swift_dgo/deepseek_gsm8k/v0-20260105-151936/checkpoint-1000 gsm8k_cot 64

# Mixtral
bash ./run_eval.sh /usr/commondata/public/hf_hub/cc/DGO/outputs/swift_sft/mixtral_gsm8k/v0-20251229-201522/checkpoint-1000 gsm8k_cot 1
bash ./run_eval.sh /usr/commondata/public/hf_hub/cc/DGO/outputs/swift_dgo/mixtral_gsm8k/v0-20260106-053721/checkpoint-1000 gsm8k_cot 1