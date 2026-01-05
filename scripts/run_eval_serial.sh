#!/bin/bash
# 串行运行实验 - 简单版
# 用法: ./run_eval_serial.sh

cd /usr/commondata/public/hf_hub/cc/DGO/scripts

bash ./run_eval.sh /usr/commondata/public/hf_hub/cc/DGO/outputs/swift_grpo/olmoe_gsm8k/v0-20260104-073639/checkpoint-1000 gsm8k_cot 64
bash ./run_eval.sh /usr/commondata/public/hf_hub/cc/DGO/outputs/swift_dgo/olmoe_gsm8k/v0-20260105-050617/checkpoint-1000 gsm8k_cot 64