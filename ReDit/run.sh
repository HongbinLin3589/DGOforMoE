export CUDA_VISIBLE_DEVICES=4
export WANDB_MODE=disabled
export HF_HOME=/data_train/kitwei/huggingface
#python grpo_gsm8k.py
#python grpo_math.py
python grpo_gsm8k_paft.py
# nohup bash run.sh > ./log/qwen2.5-7B-gsm8k-grpo-gauss-perturbed-0.01.log &
# nohup bash run.sh > ./log/qwen2.5-7B-gsm8k-grpo-perturbed-0.02.log &
# nohup bash run.sh > ./log/qwen2.5-7B-gsm8k-grpo.log &
# nohup bash run.sh > ./log/qwen2.5-7B-gsm8k-grpo-one-token.log &
# nohup bash run.sh > ./log/qwen2.5-7B-gsm8k-grpo-random-rule-5.log &
# nohup bash run.sh > ./log/qwen2.5-7B-math-grpo-gauss-0.05.log &
# nohup bash run.sh > ./log/qwen2.5-7B-grpo_gsm8k_paft.log &