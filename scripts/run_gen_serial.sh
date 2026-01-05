#!/bin/bash
# 串行运行实验 - 简单版
# 用法: ./run_gen_serial.sh

cd /usr/commondata/public/hf_hub/cc/DGO/scripts

# Gen
bash run_dgo_gen.sh deepseek gsm8k
bash run_dgo_gen.sh mixtral gsm8k

# Train
bash run_dgo_train.sh qwen gsm8k 
bash run_dgo_train.sh deepseek gsm8k
bash run_dgo_train.sh mixtral gsm8k