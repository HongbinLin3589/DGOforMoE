# ReDit
## Introduction to ReDit
DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it"s a "perfect" reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10\% the training steps, and furthermore, still exhibits a 4\% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages.

## File Structure

- **dataset**: Contains training and testing data.
- **grpo_gsm8k.py**: GSM8K training code.
- **grpo_math.py**: MATH training code.
- **run.sh**: Startup Script.


## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Now we can start training
```bash
bash run.sh  
# You need to modify the reward function of grpo_xx.py to achieve the effect of ReDit
```