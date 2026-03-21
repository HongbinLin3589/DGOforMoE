#!/bin/bash
# 串行运行实验 - 简单版
# 用法: ./run_serial.sh

cd /usr/commondata/public/hf_hub/cc/DGO/scripts

# olmoe
bash ./run_dgo_gen.sh olmoe math
bash ./run_dgo_gen.sh olmoe mbpp
bash ./run_dgo_gen.sh olmoe gsm8k

# qwen
# bash ./run_grpo_swift.sh qwen math

# deepseek
# bash ./run_sft_swift.sh deepseek gsm8k
# bash ./run_grpo_swift.sh deepseek gsm8k

# mixtral
# bash ./run_grpo_swift.sh mixtral gsm8k

echo "全部完成!"
