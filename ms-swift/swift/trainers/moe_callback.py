# Copyright (c) DGO Project
# MoE Monitor Callback - 专门针对 4 个目标模型
# =============================================================================
# 支持的模型:
# 1. OLMoE-1B-7B-0125 (Allen AI)
# 2. Qwen1.5-MoE-A2.7B (Alibaba)
# 3. deepseek-moe-16b-base (DeepSeek)
# 4. Mixtral-8x7B-v0.1 (Mistral AI)
# =============================================================================

import os
import re
import sys
import torch
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from swift.utils import get_logger

logger = get_logger()

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from moe_monitor import MoEMonitor
except ImportError:
    MoEMonitor = None


# =============================================================================
# 您的 4 个模型的精确配置
# =============================================================================

@dataclass
class MoEConfig:
    """MoE 模型配置"""
    num_experts: int           # routed 专家数
    topk: int                  # 每个 token 激活的专家数
    num_layers: int            # MoE 层数
    has_shared_expert: bool    # 是否有 shared expert
    gate_attr_name: str        # gate 模块在 MoE block 中的属性名
    config_expert_attr: str    # model.config 中专家数的属性名
    config_topk_attr: str      # model.config 中 topk 的属性名


# 精确配置 - 基于 HuggingFace 模型定义
MODEL_CONFIGS = {
    # =========================================================================
    # OLMoE-1B-7B-0125
    # 结构: model.model.layers[i].mlp.gate
    # Config: num_experts=64, num_experts_per_tok=8
    # =========================================================================
    "olmoe": MoEConfig(
        num_experts=64,
        topk=8,
        num_layers=16,
        has_shared_expert=False,
        gate_attr_name="gate",           # layers[i].mlp.gate
        config_expert_attr="num_experts",
        config_topk_attr="num_experts_per_tok",
    ),

    # =========================================================================
    # Qwen1.5-MoE-A2.7B
    # 结构: model.model.layers[i].mlp.gate (Qwen2MoE 架构)
    # Config: num_experts=60, num_experts_per_tok=4
    # 特点: 有 1 个 shared_expert + shared_expert_gate
    # =========================================================================
    "qwen": MoEConfig(
        num_experts=60,       # 60 routed experts
        topk=4,               # top-4 routing
        num_layers=24,
        has_shared_expert=True,  # 有 shared expert
        gate_attr_name="gate",   # layers[i].mlp.gate
        config_expert_attr="num_experts",
        config_topk_attr="num_experts_per_tok",
    ),

    # =========================================================================
    # deepseek-moe-16b-base
    # 结构: model.model.layers[i].mlp.gate (DeepseekMoE 架构)
    # Config: n_routed_experts=64, num_experts_per_tok=6
    # 特点: 64 routed + 2 shared experts, 28 layers
    # =========================================================================
    "deepseek": MoEConfig(
        num_experts=64,       # 64 routed experts
        topk=6,               # top-6 routing (从论文确认)
        num_layers=28,        # 28 transformer layers
        has_shared_expert=True,  # 2 shared experts
        gate_attr_name="gate",   # layers[i].mlp.gate
        config_expert_attr="n_routed_experts",
        config_topk_attr="num_experts_per_tok",
    ),

    # =========================================================================
    # Mixtral-8x7B-v0.1
    # 结构: model.model.layers[i].block_sparse_moe.gate
    # Config: num_local_experts=8, num_experts_per_tok=2
    # =========================================================================
    "mixtral": MoEConfig(
        num_experts=8,        # 8 experts
        topk=2,               # top-2 routing
        num_layers=32,
        has_shared_expert=False,
        gate_attr_name="gate",   # block_sparse_moe.gate
        config_expert_attr="num_local_experts",
        config_topk_attr="num_experts_per_tok",
    ),
}


class MoEMonitorCallback(TrainerCallback):
    """
    MoE Monitor Callback - 针对 4 个目标模型优化

    支持:
    - OLMoE-1B-7B-0125: 64 experts, top-8
    - Qwen1.5-MoE-A2.7B: 60 experts, top-4, +shared
    - deepseek-moe-16b-base: 64 experts, top-6, +2 shared
    - Mixtral-8x7B-v0.1: 8 experts, top-2

    使用方法:
    ```python
    callback = MoEMonitorCallback(
        model_key="olmoe",  # 或 "qwen", "deepseek", "mixtral"
        log_every=100,
    )
    trainer = Trainer(model=model, callbacks=[callback], ...)
    ```
    """

    def __init__(
        self,
        model_key: Optional[str] = None,  # "olmoe", "qwen", "deepseek", "mixtral"
        ref_model: Optional[torch.nn.Module] = None,
        log_every: int = 100,
        save_dir: str = "./moe_logs",
        enabled: bool = True,
        track_per_layer: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        self.enabled = enabled and MoEMonitor is not None
        if not self.enabled:
            if MoEMonitor is None:
                logger.warning("⚠️  MoEMonitor not found. MoE monitoring disabled.")
            return

        self.model_key = model_key
        self.ref_model = ref_model
        self.log_every = log_every
        self.save_dir = save_dir
        self.track_per_layer = track_per_layer
        self.verbose = verbose
        self.monitor = None

        # 配置 (将在 on_train_begin 中设置)
        self.moe_config: Optional[MoEConfig] = None
        self.detected_model: str = "unknown"

        # Hook 管理
        self.router_logits_cache: List[torch.Tensor] = []
        self.layer_indices_cache: List[int] = []
        self.hook_handles = []
        self._last_logged_step = -1
        self._latest_metrics = {}  # 存储最新的MoE指标供on_log使用

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """初始化 MoE monitor 并注册 hooks"""
        if not self.enabled or model is None:
            return

        # 设置保存目录
        self.save_dir = os.path.join(args.output_dir, "moe_logs")

        # =====================================================================
        # 检测模型类型并获取配置
        # =====================================================================
        self.moe_config, self.detected_model = self._detect_and_configure(model)

        if self.moe_config is None:
            logger.warning("⚠️  无法检测 MoE 架构，禁用监控")
            self.enabled = False
            return

        # =====================================================================
        # 初始化 monitor
        # =====================================================================
        try:
            self.monitor = MoEMonitor(
                model=model,
                ref_model=self.ref_model,
                save_dir=self.save_dir,
                num_experts=self.moe_config.num_experts,
                topk=self.moe_config.topk,
                num_layers=self.moe_config.num_layers,
            )

            if self.verbose:
                self._print_config()

        except Exception as e:
            logger.warning(f"⚠️  初始化 MoE Monitor 失败: {e}")
            self.enabled = False
            return

        # 启用 router logits 输出
        if hasattr(model, 'config'):
            model.config.output_router_logits = True

        # =====================================================================
        # 注册 hooks
        # =====================================================================
        self._register_hooks(model)

    def _detect_and_configure(self, model) -> Tuple[Optional[MoEConfig], str]:
        """检测模型类型并返回配置"""

        # 如果用户明确指定了 model_key
        if self.model_key and self.model_key in MODEL_CONFIGS:
            config = MODEL_CONFIGS[self.model_key]
            # 用实际 model.config 的值覆盖默认值
            return self._override_from_model(model, config, self.model_key), self.model_key

        # 自动检测
        model_config = getattr(model, 'config', None)
        if model_config is None:
            return None, "unknown"

        # 获取标识符
        model_type = getattr(model_config, 'model_type', '').lower()
        name_or_path = getattr(model_config, '_name_or_path', '').lower()

        # 匹配模型
        detected_key = None

        if 'olmoe' in model_type or 'olmoe' in name_or_path:
            detected_key = "olmoe"
        elif 'qwen2_moe' in model_type or 'qwen1.5-moe' in name_or_path or ('qwen' in name_or_path and 'moe' in name_or_path):
            detected_key = "qwen"
        elif 'deepseek' in model_type or 'deepseek' in name_or_path:
            detected_key = "deepseek"
        elif 'mixtral' in model_type or 'mixtral' in name_or_path:
            detected_key = "mixtral"

        if detected_key is None:
            logger.warning(f"⚠️  无法识别模型类型: model_type={model_type}, name={name_or_path}")
            return None, "unknown"

        config = MODEL_CONFIGS[detected_key]
        return self._override_from_model(model, config, detected_key), detected_key

    def _override_from_model(
        self,
        model,
        base_config: MoEConfig,
        model_key: str
    ) -> MoEConfig:
        """用 model.config 的实际值覆盖基础配置"""
        model_config = getattr(model, 'config', None)
        if model_config is None:
            return base_config

        # 读取实际值
        num_experts = getattr(model_config, base_config.config_expert_attr, base_config.num_experts)
        topk = getattr(model_config, base_config.config_topk_attr, base_config.topk)
        num_layers = getattr(model_config, 'num_hidden_layers', base_config.num_layers)

        return MoEConfig(
            num_experts=num_experts,
            topk=topk,
            num_layers=num_layers,
            has_shared_expert=base_config.has_shared_expert,
            gate_attr_name=base_config.gate_attr_name,
            config_expert_attr=base_config.config_expert_attr,
            config_topk_attr=base_config.config_topk_attr,
        )

    def _print_config(self):
        """打印配置信息"""
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ MoE Monitor 已初始化")
        logger.info(f"{'='*60}")
        logger.info(f"   模型:           {self.detected_model}")
        logger.info(f"   专家数:         {self.moe_config.num_experts}")
        logger.info(f"   Top-K:          {self.moe_config.topk}")
        logger.info(f"   层数:           {self.moe_config.num_layers}")
        logger.info(f"   Shared Expert:  {self.moe_config.has_shared_expert}")
        logger.info(f"   保存目录:       {self.save_dir}")
        logger.info(f"   日志间隔:       {self.log_every} steps")
        logger.info(f"{'='*60}\n")

    def _register_hooks(self, model):
        """注册 forward hooks 捕获 router logits"""

        # ✅ FIX: 处理 DDP wrapped model
        # 如果模型被 DistributedDataParallel 包装，需要访问内部的 module
        if hasattr(model, 'module'):
            actual_model = model.module
            if self.verbose:
                logger.info(f"   检测到 DDP 包装，使用 model.module")
        else:
            actual_model = model

        hooked_count = 0

        def make_hook(layer_idx):
            def router_hook(module, input, output):
                # ✅ FIX: 处理 tuple 输出（某些模型的 gate 返回 (logits, aux_loss)）
                if isinstance(output, tuple):
                    output = output[0]  # 通常第一个元素是 logits

                if isinstance(output, torch.Tensor) and output.dim() == 2:
                    # 验证形状: [batch*seq, num_experts]
                    if output.shape[-1] == self.moe_config.num_experts:
                        self.router_logits_cache.append(output.detach().cpu())
                        self.layer_indices_cache.append(layer_idx)
            return router_hook

        # 根据模型类型确定 gate 模块路径
        gate_patterns = self._get_gate_patterns()

        # 使用 actual_model 而不是 model
        for name, module in actual_model.named_modules():
            # 检查是否匹配 gate 模式
            if any(name.endswith(pattern) for pattern in gate_patterns):
                layer_idx = self._extract_layer_index(name)
                handle = module.register_forward_hook(make_hook(layer_idx))
                self.hook_handles.append(handle)
                hooked_count += 1
                if self.verbose and hooked_count <= 3:
                    logger.info(f"   Hook: {name} (layer {layer_idx})")

        if hooked_count > 0:
            if self.verbose:
                logger.info(f"   ✅ 注册了 {hooked_count} 个 gate hooks")
        else:
            logger.info(f"   ⚠️  未找到 gate 模块，尝试的模式: {gate_patterns}")
            self.enabled = False

    def _get_gate_patterns(self) -> List[str]:
        """根据模型类型返回 gate 模块的匹配模式"""
        patterns = {
            "olmoe": [".mlp.gate"],
            "qwen": [".mlp.gate"],
            "deepseek": [".mlp.gate"],
            "mixtral": [".block_sparse_moe.gate"],
        }
        return patterns.get(self.detected_model, [".gate", ".mlp.gate"])

    def _extract_layer_index(self, module_name: str) -> int:
        """从模块名提取层索引"""
        match = re.search(r'layers\.(\d+)', module_name)
        if match:
            return int(match.group(1))
        return -1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """在每个 step 结束后计算并记录 MoE 指标到 TensorBoard"""
        if not self.enabled or self.monitor is None:
            return

        # ✅ FIX: 判断是否需要记录，避免早期 return 导致缓存不清空（内存泄漏）
        should_log = (state.global_step % self.log_every == 0 and
                      state.global_step != self._last_logged_step)

        try:
            # ✅ 只在主进程且需要记录时才计算
            if should_log and state.is_world_process_zero and self.router_logits_cache:
                self._last_logged_step = state.global_step

                # 计算指标
                metrics = self._compute_metrics()

                if metrics:
                    # 保存最新指标供 on_log 使用（用于TensorBoard记录）
                    self._latest_metrics = metrics

                    # 保存到 monitor (只写文件，不会阻塞)
                    self.monitor.log(state.global_step, metrics)

                    # ✅ FIX: 直接记录MoE指标到logging.jsonl/TensorBoard
                    # 问题: on_log callback 未被DGO/GRPO trainer调用
                    # 解决: 直接将MoE指标添加到state.log_history，让trainer在保存logs时自动包含
                    # Note: MoE metrics are now properly logged to TensorBoard via on_log callback
                    # The callback is inserted at position 0 to ensure on_log runs before TensorBoard

                    # 每50步自动保存一次，避免训练中断丢失数据
                    if state.global_step % 50 == 0:
                        try:
                            self.monitor.save()
                        except Exception as e:
                            logger.warning(f"⚠️  自动保存 MoE 指标失败: {e}")

                    # 打印摘要
                    metric_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                    logger.info(f"\n📊 [Step {state.global_step}] MoE ({self.detected_model}): {metric_str}")

                    # 警告
                    if metrics.get('collapse_rate', 0) > 0.3:
                        logger.info(f"   ⚠️  高崩溃率: {metrics['collapse_rate']:.1%}")
                    if metrics.get('load_cv', 0) > 1.0:
                        logger.info(f"   ⚠️  高负载不均衡: CV={metrics['load_cv']:.2f}")

        except Exception as e:
            logger.warning(f"⚠️  计算 MoE 指标时出错 (step {state.global_step}): {e}")
            import traceback
            traceback.print_exc()
        finally:
            # ✅ FIX: 确保总是清空缓存（修复内存泄漏和错误恢复）
            # 所有进程都需要清空缓存，避免内存积累
            self.router_logits_cache = []
            self.layer_indices_cache = []

    def _compute_metrics(self) -> Dict[str, float]:
        """计算所有 4 个指标（使用 bincount 优化）"""
        if not self.router_logits_cache:
            return {}

        try:
            metrics = {}

            # 计算 Load CV
            avg_cv, layer_cvs = self._compute_load_cv()
            metrics['load_cv'] = avg_cv

            if self.track_per_layer and layer_cvs:
                for layer_idx, cv in layer_cvs.items():
                    metrics[f'layer_{layer_idx}_cv'] = cv

            # 计算崩溃率
            metrics['collapse_rate'] = self._compute_collapse_rate()

            # 计算路由熵 (Routing Entropy)
            metrics['routing_entropy'] = self._compute_routing_entropy()

            # 计算最大负载 (Max Load)
            metrics['max_load'] = self._compute_max_load()

            return metrics

        except Exception as e:
            logger.warning(f"⚠️  _compute_metrics 出错: {e}")
            return {}

    def _compute_load_cv(self) -> Tuple[float, Dict[int, float]]:
        """计算负载变异系数

        修复：先按层累积所有 batch 的 expert counts，再计算 CV

        之前的 bug：每个 batch 单独算 CV 再平均，会导致 CV 被人为放大
        例如：batch1 专家0负载高(CV=1.5)，batch2 专家1负载高(CV=1.5)
              错误方法: avg_cv = 1.5
              正确方法: 合并后均衡，cv = 0.2
        """
        from collections import defaultdict

        # Step 1: 按层累积所有 batch 的 expert counts
        layer_counts = defaultdict(lambda: torch.zeros(self.moe_config.num_experts))

        for logits, layer_idx in zip(self.router_logits_cache, self.layer_indices_cache):
            # logits: [N, num_experts]
            topk_indices = torch.topk(logits, self.moe_config.topk, dim=-1).indices
            flat_indices = topk_indices.flatten()

            # 使用 bincount 计数 (比循环快 100x)
            expert_counts = torch.bincount(
                flat_indices,
                minlength=self.moe_config.num_experts
            ).float()

            # ✅ 累积到同一层（而不是每个batch单独计算）
            layer_counts[layer_idx] += expert_counts

        # Step 2: 每层计算一次 CV
        layer_cvs = {}
        for layer_idx, counts in layer_counts.items():
            total = counts.sum()
            if total == 0:
                continue

            load = counts / total
            cv = (load.std() / (load.mean() + 1e-8)).item()
            layer_cvs[layer_idx] = cv

        # Step 3: 返回所有层的平均 CV
        avg_cv = sum(layer_cvs.values()) / len(layer_cvs) if layer_cvs else 0.0

        return avg_cv, layer_cvs

    def _compute_collapse_rate(self, threshold: float = 0.01) -> float:
        """计算路由崩溃率

        Args:
            threshold: 绝对阈值，负载 < threshold 的专家被认为"崩溃"
                      默认 0.01 (1%)，即负载低于 1% 的专家算崩溃

        修复：按层累积后计算每层的崩溃率，返回最大值（最差情况）
        之前的 bug：混合所有层，某层崩溃会被其他层掩盖
        """
        from collections import defaultdict

        # Step 1: 按层累积 counts
        layer_counts = defaultdict(lambda: torch.zeros(self.moe_config.num_experts))

        for logits, layer_idx in zip(self.router_logits_cache, self.layer_indices_cache):
            topk_indices = torch.topk(logits, self.moe_config.topk, dim=-1).indices
            flat_indices = topk_indices.flatten()

            expert_counts = torch.bincount(
                flat_indices,
                minlength=self.moe_config.num_experts
            ).float()

            layer_counts[layer_idx] += expert_counts

        # Step 2: 每层计算崩溃率
        layer_collapse_rates = []
        for counts in layer_counts.values():
            total = counts.sum()
            if total == 0:
                continue

            load = counts / total
            collapsed = (load < threshold).sum().item()
            collapse_rate = collapsed / self.moe_config.num_experts
            layer_collapse_rates.append(collapse_rate)

        # Step 3: 返回最大崩溃率（反映最差的层）
        return max(layer_collapse_rates) if layer_collapse_rates else 0.0

    def _compute_routing_entropy(self) -> float:
        """
        计算路由熵 (Routing Entropy)

        熵越高表示路由分布越均匀，越低表示越集中
        理想情况下: entropy = log(num_experts) 表示完全均匀

        修复：按层累积后计算每层的熵，再取平均
        之前的 bug：混合所有层，某层崩溃会被其他层掩盖
        """
        from collections import defaultdict

        # Step 1: 按层累积 counts
        layer_counts = defaultdict(lambda: torch.zeros(self.moe_config.num_experts))

        for logits, layer_idx in zip(self.router_logits_cache, self.layer_indices_cache):
            topk_indices = torch.topk(logits, self.moe_config.topk, dim=-1).indices
            flat_indices = topk_indices.flatten()

            expert_counts = torch.bincount(
                flat_indices,
                minlength=self.moe_config.num_experts
            ).float()

            layer_counts[layer_idx] += expert_counts

        # Step 2: 每层计算熵
        max_entropy = torch.log(torch.tensor(float(self.moe_config.num_experts))).item()
        layer_entropies = []

        for counts in layer_counts.values():
            total = counts.sum()
            if total == 0:
                continue

            probs = counts / total
            probs = probs[probs > 0]

            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            normalized = entropy / max_entropy if max_entropy > 0 else 0.0
            layer_entropies.append(normalized)

        # Step 3: 返回平均熵
        return sum(layer_entropies) / len(layer_entropies) if layer_entropies else 0.0

    def _compute_max_load(self) -> float:
        """
        计算最大专家负载

        表示最繁忙专家承担的 token 比例
        理想情况下: max_load = 1/num_experts (完全均衡)
        值越高表示负载越不均衡

        修复：按层累积后计算每层的 max_load，返回最大值（最差情况）
        之前的 bug：混合所有层后取 max，峰值被平滑了
        """
        from collections import defaultdict

        # Step 1: 按层累积 counts
        layer_counts = defaultdict(lambda: torch.zeros(self.moe_config.num_experts))

        for logits, layer_idx in zip(self.router_logits_cache, self.layer_indices_cache):
            topk_indices = torch.topk(logits, self.moe_config.topk, dim=-1).indices
            flat_indices = topk_indices.flatten()

            expert_counts = torch.bincount(
                flat_indices,
                minlength=self.moe_config.num_experts
            ).float()

            layer_counts[layer_idx] += expert_counts

        # Step 2: 每层计算 max_load，取最大值（最差情况）
        layer_max_loads = []
        for counts in layer_counts.values():
            total = counts.sum()
            if total == 0:
                continue

            load = counts / total
            layer_max_loads.append(load.max().item())

        # Step 3: 返回所有层中的最大值（反映最不均衡的层）
        return max(layer_max_loads) if layer_max_loads else 0.0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs
    ):
        """
        将MoE指标添加到trainer的日志中，使其记录到TensorBoard。

        注意：on_log在所有进程同步后调用，修改logs字典是安全的。
        """
        if not self.enabled or logs is None:
            return

        # 只在主进程添加指标
        if not state.is_world_process_zero:
            return

        # 将最新的MoE指标添加到logs，TensorBoard会自动记录
        if hasattr(self, '_latest_metrics') and self._latest_metrics:
            for key, value in self._latest_metrics.items():
                # 添加前缀以便在TensorBoard中分组显示
                logs[f'moe/{key}'] = value

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """清理并保存"""
        if not self.enabled:
            return

        # 移除 hooks (所有进程都需要执行)
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.router_logits_cache = []
        self.layer_indices_cache = []

        # 只在主进程保存指标和打印日志
        if not state.is_world_process_zero:
            return

        if self.monitor is not None:
            try:
                self.monitor.save()

                if self.verbose:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"✅ MoE 训练完成 - {self.detected_model}")
                    logger.info(f"{'='*60}")
                    logger.info(f"   指标已保存到: {self.save_dir}/moe_metrics.json")

                    if self.monitor.history.get('step'):
                        for metric in ['load_cv', 'collapse_rate']:
                            if metric in self.monitor.history:
                                values = self.monitor.history[metric]
                                if values:
                                    logger.info(f"   {metric}: min={min(values):.4f}, "
                                          f"max={max(values):.4f}, final={values[-1]:.4f}")
                    logger.info(f"{'='*60}\n")

            except Exception as e:
                logger.warning(f"⚠️  保存 MoE 指标失败: {e}")


# =============================================================================
# 便捷函数
# =============================================================================

def get_moe_callback(
    model_key: Optional[str] = None,
    log_every: int = 100,
    **kwargs
) -> MoEMonitorCallback:
    """
    获取 MoE 监控 Callback

    Args:
        model_key: 模型标识符
            - "olmoe": OLMoE-1B-7B-0125
            - "qwen": Qwen1.5-MoE-A2.7B
            - "deepseek": deepseek-moe-16b-base
            - "mixtral": Mixtral-8x7B-v0.1
            - None: 自动检测
        log_every: 日志间隔 (steps)

    Example:
        ```python
        callback = get_moe_callback("olmoe", log_every=100)
        trainer = Trainer(model=model, callbacks=[callback], ...)
        ```
    """
    return MoEMonitorCallback(model_key=model_key, log_every=log_every, **kwargs)


# =============================================================================
# 向后兼容别名
# =============================================================================
UniversalMoECallback = MoEMonitorCallback
MoEMonitorCallbackV3 = MoEMonitorCallback
MoEMonitorCallbackV4 = MoEMonitorCallback


# =============================================================================
# 配置速查表
# =============================================================================
"""
模型配置速查表:

| 模型                  | 专家数 | Top-K | 层数 | Shared | Gate 路径                    |
|-----------------------|--------|-------|------|--------|------------------------------|
| OLMoE-1B-7B-0125      | 64     | 8     | 16   | ❌     | layers[i].mlp.gate           |
| Qwen1.5-MoE-A2.7B     | 60     | 4     | 24   | ✅ (1) | layers[i].mlp.gate           |
| deepseek-moe-16b-base | 64     | 6     | 28   | ✅ (2) | layers[i].mlp.gate           |
| Mixtral-8x7B-v0.1     | 8      | 2     | 32   | ❌     | layers[i].block_sparse_moe.gate |

Config 属性对照:
- OLMoE:    config.num_experts=64, config.num_experts_per_tok=8
- Qwen:     config.num_experts=60, config.num_experts_per_tok=4
- DeepSeek: config.n_routed_experts=64, config.num_experts_per_tok=6
- Mixtral:  config.num_local_experts=8, config.num_experts_per_tok=2
"""
