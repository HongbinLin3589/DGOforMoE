# =================================================================================
# 配置HuggingFace缓存目录和镜像源（必须在所有导入之前）
# =================================================================================
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

# ========== LoRA 架构检测函数 ==========
def get_lora_target_modules(model_name: str) -> List[str]:
    """
    根据模型架构自动选择 LoRA 目标模块
    仅对注意力模块应用 LoRA，不对 MoE 路由器应用
    对路由器应用 LoRA 会破坏负载均衡机制和 auxiliary loss 计算
    """
    # 所有模型：只对注意力投影层应用 LoRA
    # 不包括 'gate' (MoE 路由器) 以保护负载均衡机制
    return ["q_proj", "k_proj"]

# ========== 多卡训练：GPU 检测和梯度累积动态计算 ==========
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    print("⚠️  Warning: No GPU detected, defaulting to 8")
    num_gpus = 8

# 梯度累积动态计算（保持全局批大小 = 256）
global_batch_size = 256
per_device_train_batch_size = 8
gradient_accumulation_steps = global_batch_size // (num_gpus * per_device_train_batch_size)

print(f">>> GPUs detected: {num_gpus}")
print(f">>> Per-device batch size: {per_device_train_batch_size}")
print(f">>> Gradient accumulation steps: {gradient_accumulation_steps}")
print(f">>> Global batch size: {num_gpus * per_device_train_batch_size * gradient_accumulation_steps}")

# Load and prep dataset




# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted response."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# uncomment middle messages for 1-shot prompting
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
def get_math_questions() -> Dataset:
    #data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = load_dataset('parquet', data_files=f'/usr/commondata/public/hf_hub/cc/DGO/ReDit/dataset/MATH/data/train/0000.parquet')['train']
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'user', 'content': x['problem']+ " " + instruction_following}
        ],
        'answer': extract_solution(x['solution'])
    }) 
    return data 

dataset = get_math_questions()


def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 2.0
    except Exception as e:
        print(e)

    return retval

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}")
    return [compute_score(r, a) for r, a in zip(responses, answer)]



def reasoning_steps_reward(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]

def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.25
        if text.count("\n</reasoning>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]

def len_reward(completions, answer, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    def extract_number(s: str) -> str:
        """提取字符串中的数字部分"""
        match = re.search(r'\d+', s)
        return match.group(0) if match else ''
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    original_rewards = [2.0 if extract_number(r) == extract_number(a) else 0.0 for r, a in zip(extracted_responses, answer)]

    # Calculate lengths
    lengths = [len(content) for content in responses]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, acc_reward in zip(lengths, original_rewards):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if acc_reward >= 1.0:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards



def cosine_scaled_reward(completions, answer, 
                                       cosine_min_len_value_wrong: float = 0.0,
                                       cosine_max_len_value_wrong: float = -0.5,
                                       cosine_min_len_value_correct: float = 1.0,
                                       cosine_max_len_value_correct: float = 0.5,
                                       cosine_max_len: int = 1000, **kwargs) -> list[float]:
    # https://arxiv.org/abs/2502.03373
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def extract_number(s: str) -> str:
        """提取字符串中的数字部分"""
        match = re.search(r'\d+', s)
        return match.group(0) if match else ''
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # 仅判断数字部分是否相等
    original_rewards = [2.0 if extract_number(r) == extract_number(a) else 0.0 for r, a in zip(extracted_responses, answer)]
    
    rewards = []
    for content, acc_reward in zip(responses, original_rewards):
        is_correct = acc_reward >= 1.0
        if is_correct:
            # Swap min/max for correct answers
            min_value = cosine_max_len_value_correct
            max_value = cosine_min_len_value_correct
        else:
            min_value = cosine_min_len_value_wrong
            max_value = cosine_max_len_value_wrong
        
        gen_len = len(content)
        reward = cosfn(gen_len, cosine_max_len, min_value, max_value)
        rewards.append(reward)
    
    return rewards

def repetition_penalty_reward(completions, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0, **kwargs) -> list[float]:
    """
    Reward function that penalizes repetitions in the completions.

    Args:
        completions: List of model completions
        repetition_n_grams: The size of n-grams to consider for repetition
        repetition_max_penalty: The maximum penalty for repetitions

    Returns:
        List of float rewards based on the repetition penalty.
    """
    
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    for completion in responses:
        #print(f"completion : {completion}")
        if completion == '':
            rewards.append(0.0)
            continue
        if len(completion.split()) < repetition_n_grams:
            rewards.append(0.0)
            continue

        ngrams = set()
        total = 0
        for ng in zipngram(completion, repetition_n_grams):
            ngrams.add(ng)
            total += 1

        scaling = 1 - len(ngrams) / total
        reward = scaling * repetition_max_penalty
        rewards.append(reward)
    
    return rewards

# Reward functions with noise
# noise 固定， 0.1, 0.3, 0.5
m = 0.05
print(f"noise: {m}")


def correctness_reward_func_with_noise(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}")
    original_rewards = [compute_score(r, a) for r, a in zip(responses, answer)]
    #noisy_rewards = [r + random.uniform(-m * 2.0, m * 2.0) for r in original_rewards]
    noisy_rewards = [r + random.gauss(0, 2.0 * m  / (3 ** 0.5))  for r in original_rewards]
    return noisy_rewards



def reasoning_steps_reward_with_noise(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]
    original_rewards = [min(1.0, count / 3) for count in matches]
    #noisy_rewards = [r + random.uniform(-m * 1.0, m * 1.0) for r in original_rewards]
    noisy_rewards = [r + random.gauss(0, 1.0 * m  / (3 ** 0.5))  for r in original_rewards]
    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return noisy_rewards






# Reward functions with scheduled noise
def SquareRootNoise(noise, num_step):
    return noise * math.pow(num_step / 300 + 1.0, -0.5)

def SquareRootReverseNoise(noise, num_step, max_step):
    return SquareRootNoise(noise, max_step - num_step)

def FactorNoise(noise, num_step, factor, stop_factor_noise):
    return max(stop_factor_noise, noise * (factor ** num_step))

def FactorReverseNoise(noise, num_step, factor, stop_factor_noise, max_step):
    return FactorNoise(noise, max_step - num_step, factor, stop_factor_noise)

import bisect
def MutilFactorNoise(noise, step, factor, num_step):
    index = bisect.bisect_right(step, num_step)
    noise *= (factor ** index)
    return noise

def MutilFactorReverseNoise(noise, step, factor, num_step, max_step):
    return MutilFactorNoise(noise, step, factor, max_step - num_step)


def Cosine_scheduler(num_step, max_update, base_noise, final_noise, warmup_steps, warmup_begin_noise):
    if warmup_steps > 0 and num_step < warmup_steps:
        increase = (base_noise - warmup_begin_noise) * float(num_step) / float(warmup_steps)
        return warmup_begin_noise + increase
    if num_step <= max_update:
        max_steps = max_update - warmup_steps
        noise = final_noise + (base_noise - final_noise) * (1 + math.cos(math.pi * (num_step - warmup_steps) / max_steps)) / 2
        return noise
    return final_noise

def CosineReverse_scheduler(num_step, max_update, base_noise, final_noise, warmup_steps, warmup_begin_noise, max_step):
    return Cosine_scheduler(num_step = max_step - num_step, max_update=max_update, base_noise=base_noise, final_noise=final_noise, warmup_steps=warmup_steps, warmup_begin_noise=warmup_begin_noise)

    
def scheduler(original_rewards, num_step, max_step):
    #perturbation_noise = SquareRootNoise(noise = 0.1, num_step = num_step)
    #perturbation_noise = SquareRootReverseNoise(noise = 0.1, num_step = num_step, max_step = max_step)
    #perturbation_noise = FactorNoise(noise = 0.1, num_step = num_step, factor = 0.99, stop_factor_noise = 1e-3)
    #perturbation_noise = FactorReverseNoise(noise = 0.1, num_step = num_step, factor = 0.99, stop_factor_noise = 1e-3, max_step = max_step)
    #perturbation_noise = MutilFactorNoise(noise = 0.1, step = [2000, 4000, 6000, 8000],  factor = 0.5, num_step = num_step)
    #perturbation_noise = MutilFactorReverseNoise(noise = 0.1, step = [2000, 4000, 6000, 8000],  factor = 0.5, num_step = num_step, max_step = max_step)
    #perturbation_noise = Cosine_scheduler(num_step = num_step, max_update=8000, base_noise=0.1, final_noise=1e-3, warmup_steps=1000, warmup_begin_noise=0)
    perturbation_noise = CosineReverse_scheduler(num_step = num_step, max_update=8000, base_noise=0.1, final_noise=1e-3, warmup_steps=0, warmup_begin_noise=0, max_step = max_step)
    print("perturbation_noise", perturbation_noise)
    return random.gauss(0, original_rewards * perturbation_noise / (3 ** 0.5))

def correctness_reward_func_with_scheduler_noise(prompts, completions, answer, global_step, max_step, **kwargs) -> list[float]:
    def extract_number(s: str) -> str:
        """提取字符串中的数字部分"""
        match = re.search(r'\d+', s)
        return match.group(0) if match else ''
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    
    # 仅判断数字部分是否相等
    original_rewards = [2.0 if extract_number(r) == extract_number(a) else 0.0 for r, a in zip(extracted_responses, answer)]
    
    noisy_rewards = [r + scheduler(2.0, global_step, max_step) for r in original_rewards] 

    return noisy_rewards

def int_reward_func_with_scheduler_noise(completions, global_step, max_step, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Calculate the original rewards
    original_rewards = [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
    
    noisy_rewards = [r + scheduler(0.5, global_step, max_step) for r in original_rewards] 
    
    return noisy_rewards

def strict_format_reward_func_with_scheduler_noise(completions, global_step, max_step, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n[\s\S]*?\n</reasoning>\n<answer>\n[\s\S]*?</answer>$"
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    
    # Calculate the original rewards
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    original_rewards = [1.0 if match else 0.0 for match in matches]
    
    noisy_rewards = [r + scheduler(1.0, global_step, max_step) for r in original_rewards] 

    return noisy_rewards

def soft_format_reward_func_with_scheduler_noise(completions, global_step, max_step, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>[\s\S]*?</reasoning>[\s\S]*?<answer>[\s\S]*?</answer>$"
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    
    # Calculate the original rewards
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    original_rewards = [1.0 if match else 0.0 for match in matches]
    
    noisy_rewards = [r + scheduler(1.0, global_step, max_step) for r in original_rewards] 
    
    return noisy_rewards

def xmlcount_reward_func_with_scheduler_noise(completions, global_step, max_step, **kwargs) -> list[float]:
    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            #count -= len(text.split("\n</answer>\n")[-1])*0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count
    contents = [completion[0]["content"] for completion in completions]
    
    # Calculate the original rewards
    original_rewards = [count_xml(c) for c in contents]
    
    noisy_rewards = [r + scheduler(0.5, global_step, max_step) for r in original_rewards] 

    return noisy_rewards




#model_name = "/data_train/kitwei/model/Llama-3.2-1B-Instruct"k
#model_name = "/data_train/kitwei/model/Qwen2.5-1.5B-Instruct"
#model_name = "/data_train/kitwei/model/Qwen2.5-0.5B-Instruct"
model_name = "/data_train/kitwei/model/Qwen2.5-7B-Instruct"
if "Llama" in model_name:
    output_dir = "outputs/group_b_grpo_llama_math"
    run_name = "group_b_grpo_llama_math"
elif '1.5B' in model_name:
    output_dir="outputs/group_b_grpo_qwen-1.5b_math"
    run_name="group_b_grpo_qwen-1.5b_math"
elif '7B' in model_name:
    output_dir="outputs/group_b_grpo_qwen-7b_math"
    run_name="group_b_grpo_qwen-7b_math"
else:
    output_dir="outputs/group_b_grpo_qwen-0.5b_math"
    run_name="group_b_grpo_qwen-0.5b_math"
    
training_args = GRPOConfig(
    #use_vllm = True,
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=1024,
    max_steps=1000,
    save_steps=100,
    eval_steps=50,
    max_grad_norm=0.1,
    #report_to="wandb",
    log_on_each_node=False,
)


                           
target_modules = get_lora_target_modules(model_name)
peft_config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=target_modules,
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
print(f">>> LoRA target modules: {target_modules}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
#cosine_scaled_reward,
#reasoning_steps_reward

# trainer = GRPOTrainer(
#     model=model,
#     processing_class=tokenizer,
#     reward_funcs=[
#         reasoning_steps_reward,
#         correctness_reward_func,
#         ],
#     args=training_args,
#     train_dataset=dataset,
#     peft_config=peft_config
# )


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reasoning_steps_reward_with_noise,
        correctness_reward_func_with_noise
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)
trainer.train()