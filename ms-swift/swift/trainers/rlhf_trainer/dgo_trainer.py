# Copyright (c) DGO Project
# DGO Trainer - Offline Weighted SFT for Distribution-Guided Optimization
# =============================================================================
#
# DGO is an offline RL approach that decouples generation and optimization:
# 1. Generation Phase: Generate N responses per prompt, compute rewards and DGO weights
# 2. Training Phase: Offline weighted SFT using pre-computed weights
#
# Usage:
#   swift rlhf --rlhf_type dgo --model ... --dgo_data_file data.json --reward_funcs accuracy
# =============================================================================

import inspect
import json
import logging
import torch
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from torch import nn
from datasets import Dataset
from transformers import PreTrainedModel, Trainer

from swift.plugin import orms
from swift.trainers.mixin import SwiftMixin

# Import trainers to apply OLMoE load_balancing_loss_func patch for DeepSpeed ZeRO compatibility
from swift.trainers import trainers as _  # noqa: F401  # Trigger module-level patch execution

logger = logging.getLogger(__name__)


def compute_dgo_weights(rewards: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """
    Compute DGO weights from rewards.

    Args:
        rewards: Tensor of shape [num_generations] or [batch_size, num_generations]
        beta: Temperature parameter (lower = sharper distribution)

    Returns:
        weights: Tensor of same shape as rewards

    Formula:
        w[i] = exp(r[i] / beta) / mean(exp(r / beta))
    """
    exp_rewards = torch.exp(rewards / beta)

    if rewards.ndim == 2:
        z_hat = exp_rewards.mean(dim=1, keepdim=True)
    else:
        z_hat = exp_rewards.mean()

    weights = exp_rewards / z_hat
    return weights


class DGOTrainer(SwiftMixin, Trainer):
    """
    DGO Trainer - Offline Weighted SFT

    Key differences from standard Trainer:
    1. Custom data collator (adds weights)
    2. Custom compute_loss (weighted cross-entropy)
    3. No online generation (uses pre-generated data)

    Follows the same pattern as GRPOTrainer for reward function handling.
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,  # ✅ 添加 ref_model 参数（可选）
        reward_funcs: Optional[List[Union[str, Callable]]] = None,
        *_args,
        **kwargs,
    ):
        """
        Initialize DGO Trainer.

        Args:
            model: The model to train
            ref_model: Reference model for MoE monitoring (optional).
                      Typically the SFT model used for DGO data generation.
                      If provided, enables routing_kl and saturation metrics.
            reward_funcs: List of reward function names (e.g., ['accuracy']) or callables
            **kwargs: Additional arguments passed to Trainer
        """
        from swift.trainers.rlhf_arguments import DGOConfig

        args: DGOConfig = kwargs['args']
        self.args = args
        self.dgo_beta = args.dgo_beta
        # Use max_length first, fallback to max_completion_length
        self.max_length = getattr(args, 'max_length', None) or getattr(args, 'max_completion_length', 1024)
        self.freeze_router = getattr(args, 'freeze_router', False)

        # ✅ 保存 ref_model（可选，默认为 None，不影响训练）
        self.ref_model = ref_model

        # Get tokenizer/processing_class from template (same as SwiftMixin)
        template = kwargs.get('template')
        if template is not None:
            self.processing_class = template.tokenizer
        else:
            self.processing_class = kwargs.get('processing_class') or kwargs.get('tokenizer')
        if self.processing_class is None:
            raise ValueError("Must provide 'template' with tokenizer, or 'processing_class'/'tokenizer'")

        # Ensure tokenizer has pad token
        if self.processing_class.pad_token is None:
            self.processing_class.pad_token = self.processing_class.eos_token

        # Prepare reward functions (same pattern as GRPOTrainer)
        self._prepare_rewards(reward_funcs or args.reward_funcs or [])

        # Load and process dataset
        if args.dgo_data_file:
            logger.info(f"Loading DGO data from: {args.dgo_data_file}")
            train_dataset = self._load_dgo_dataset(args.dgo_data_file)
            kwargs['train_dataset'] = train_dataset

        # Set custom data collator
        kwargs['data_collator'] = self._dgo_data_collator

        super().__init__(model=model, *_args, **kwargs)

        # Freeze router parameters if requested (for Group D experiments)
        if self.freeze_router:
            self._freeze_router_params(model)

        # Add MoE monitoring callback if enabled
        if getattr(args, 'moe_monitor_enabled', False):
            self._add_moe_monitor_callback()

    def _freeze_router_params(self, model: nn.Module):
        """
        Freeze all router/gate parameters for Group D experiments.
        This ensures routing consistency with the reference model.
        """
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'gate' in name.lower() or 'router' in name.lower():
                param.requires_grad = False
                frozen_count += 1
                logger.debug(f"Froze router parameter: {name}")
        logger.info(f"Froze {frozen_count} router parameters (freeze_router=True)")

    def _add_moe_monitor_callback(self):
        """
        Add MoE monitoring callback if enabled.

        Note: If ref_model is provided, can log all 5 metrics including routing_kl and saturation.
              If ref_model is None, can only log 3 basic metrics (load_cv, collapse_rate, switching_freq).
        """
        # Check if MoE monitoring is enabled
        if not getattr(self.args, 'moe_monitor_enabled', False):
            return

        try:
            from swift.trainers.moe_callback import MoEMonitorCallback

            moe_callback = MoEMonitorCallback(
                ref_model=self.ref_model,  # ✅ 使用 self.ref_model（可选）
                log_every=getattr(self.args, 'moe_log_every', 100),
                save_dir=getattr(self.args, 'moe_save_dir', None),
                enabled=True
            )
            # ✅ FIX: 插入到callback列表开头，确保on_log在TensorBoard之前执行
            self.callback_handler.callbacks.insert(0, moe_callback)

            # ✅ 根据是否有 ref_model 打印不同的信息
            if self.ref_model is not None:
                logger.info("✅ MoE monitoring callback added to DGO trainer (with ref_model)")
                logger.info("   Available metrics: load_cv, collapse_rate, switching_freq, routing_kl, saturation")
            else:
                logger.info("✅ MoE monitoring callback added to DGO trainer (without ref_model)")
                logger.info("   Available metrics: load_cv, collapse_rate, switching_freq")
                logger.info("   💡 Tip: Provide ref_model to DGOTrainer for routing_kl and saturation metrics")

        except ImportError as e:
            logger.warning(f"⚠️  Failed to import MoEMonitorCallback: {e}")
            logger.warning("   MoE monitoring will be disabled. Make sure moe_monitor.py is in the project root.")

    def _prepare_rewards(self, reward_funcs: List[Union[str, Callable]]):
        """
        Prepare reward functions, following the same pattern as GRPOTrainer.

        Args:
            reward_funcs: List of reward function names (from orms) or callables
        """
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs] if reward_funcs else []

        prepared_funcs = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                # Look up in orms (same as GRPOTrainer)
                if reward_func in orms:
                    reward_func_class = orms[reward_func]
                    reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                    reward_func_kwargs = {
                        key: getattr(self.args, key)
                        for key in reward_func_args
                        if key not in ['self', 'args', 'kwargs'] and hasattr(self.args, key)
                    }
                    if 'tokenizer' in reward_func_args:
                        reward_func_kwargs['tokenizer'] = self.processing_class
                    prepared_funcs.append(reward_func_class(**reward_func_kwargs))
                else:
                    raise ValueError(f"Reward function '{reward_func}' not found in swift.plugin.orms")
            elif callable(reward_func):
                prepared_funcs.append(reward_func)
            else:
                raise ValueError(f"Invalid reward function type: {type(reward_func)}")

        self.reward_funcs = prepared_funcs
        self.reward_func_names = []
        for reward_func in self.reward_funcs:
            if inspect.isfunction(reward_func):
                self.reward_func_names.append(reward_func.__name__)
            else:
                self.reward_func_names.append(reward_func.__class__.__name__)

        logger.info(f"DGO reward functions: {self.reward_func_names}")

    def _load_dgo_dataset(self, data_file: str) -> Dataset:
        """
        Load DGO data and convert to HuggingFace Dataset.

        Supports two formats:
        1. Grouped: [{"prompt", "completions"/"generated_texts", "weights", ...}]
        2. Flattened: [{"prompt", "response", "weight"}]
        """
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        flat_samples = []

        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]

            # Format 1: Grouped (gfol.py / vllm_inference.py format)
            if 'completions' in first_item or 'generated_texts' in first_item:
                logger.info("Detected grouped format, flattening...")

                for item in data:
                    prompt = item.get('prompt', '')
                    completions = item.get('completions') or item.get('generated_texts', [])
                    weights = item.get('weights')
                    ground_truth = item.get('ground_truth') or item.get('ground_truth_answer')

                    # Compute weights if needed
                    recompute = getattr(self.args, 'dgo_recompute_weights', False)
                    if recompute or weights is None:
                        if self.reward_funcs and ground_truth:
                            rewards = self._compute_rewards(
                                [prompt] * len(completions),
                                completions,
                                [ground_truth] * len(completions)
                            )
                            weights = compute_dgo_weights(rewards, self.dgo_beta).tolist()
                        else:
                            # Default: uniform weights
                            weights = [1.0] * len(completions)

                    for completion, weight in zip(completions, weights):
                        flat_samples.append({
                            'prompt': prompt,
                            'response': completion,
                            'weight': float(weight)
                        })

            # Format 2: Already flattened
            elif 'response' in first_item:
                logger.info("Detected flattened format")
                flat_samples = data

            else:
                raise ValueError(f"Unknown data format. Keys: {first_item.keys()}")

        # Convert to Dataset
        dataset = Dataset.from_list(flat_samples)

        # Log statistics
        weights = torch.tensor([s['weight'] for s in flat_samples])
        logger.info(f"DGO Dataset Statistics:")
        logger.info(f"  Total samples: {len(dataset)}")
        logger.info(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        logger.info(f"  Weight mean: {weights.mean():.3f}")
        logger.info(f"  Weight std: {weights.std():.3f}")

        return dataset

    def _compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        solutions: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards using registered reward functions.
        Same pattern as GRPOTrainer.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings (model outputs)
            solutions: List of ground truth solutions (for accuracy reward)

        Note: Uses 'solution' parameter name to match ms-swift ORM interface
              (e.g., MathAccuracy expects 'solution' not 'answer')
        """
        rewards = torch.zeros(len(completions))
        for reward_func in self.reward_funcs:
            try:
                # Call reward function with ms-swift standard interface
                # ORM functions expect: completions, solution, **kwargs
                r = reward_func(completions=completions, solution=solutions)
                if isinstance(r, list):
                    r = torch.tensor(r)
                rewards += r
            except Exception as e:
                logger.warning(f"Reward function {reward_func} failed: {e}")
        return rewards

    def _dgo_data_collator(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with weights and proper prompt masking.

        Args:
            features: List of dicts with {'prompt', 'response', 'weight'}

        Returns:
            Dict with {'input_ids', 'attention_mask', 'labels', 'weights'}
        """
        # Concatenate prompt + response (avoid duplicate EOS token)
        eos_token = self.processing_class.eos_token
        texts = []
        for f in features:
            text = f['prompt'] + f['response']
            # Only add EOS if not already present
            if eos_token and not text.endswith(eos_token):
                text += eos_token
            texts.append(text)
        weights = torch.tensor([f['weight'] for f in features], dtype=torch.float32)

        # Tokenize
        tokenized = self.processing_class(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Create labels (copy of input_ids)
        labels = tokenized['input_ids'].clone()

        # Mask prompt tokens (set to -100)
        for i, feature in enumerate(features):
            # Tokenize prompt separately to get its length
            prompt_tokens = self.processing_class(
                feature['prompt'],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length
            )
            prompt_length = len(prompt_tokens['input_ids'])

            # Mask prompt tokens
            labels[i, :prompt_length] = -100

        # Mask padding tokens
        labels[labels == self.processing_class.pad_token_id] = -100

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
            'weights': weights,
        }

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute weighted cross-entropy loss.

        Key difference from standard SFT:
        - Loss is weighted by DGO weights
        - High-reward samples contribute more to gradient
        """
        # Pop weights (not needed by model)
        weights = inputs.pop('weights').to(model.device)

        # Move inputs to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Per-token loss (no reduction)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Reshape to [batch_size, seq_len-1]
        loss_per_sample = loss.view(shift_labels.size())

        # Average per sample (only over non-masked tokens)
        mask = (shift_labels != -100).float()
        sum_loss = (loss_per_sample * mask).sum(dim=1)
        num_tokens = mask.sum(dim=1).clamp(min=1.0)
        avg_loss = sum_loss / num_tokens

        # Weight and average across batch
        weighted_loss = (avg_loss * weights).mean()

        # Log statistics periodically
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'dgo/unweighted_loss': avg_loss.mean().item(),
                'dgo/weighted_loss': weighted_loss.item(),
                'dgo/weight_mean': weights.mean().item(),
                'dgo/weight_std': weights.std().item(),
            })

        if return_outputs:
            return weighted_loss, outputs
        return weighted_loss


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DGO Trainer Module Test")
    print("=" * 60)

    # Test weight computation
    rewards = torch.tensor([0.0, 1.0, 2.0, 0.5])
    weights = compute_dgo_weights(rewards, beta=0.1)
    print(f"\nTest compute_dgo_weights:")
    print(f"  Rewards: {rewards.tolist()}")
    print(f"  Weights: {[f'{w:.3f}' for w in weights.tolist()]}")
    print(f"  Sum: {weights.sum():.3f}")

    print("\n" + "=" * 60)
    print("DGO Trainer ready for use!")
    print("Usage: swift rlhf --rlhf_type dgo --reward_funcs accuracy ...")
    print("=" * 60)
