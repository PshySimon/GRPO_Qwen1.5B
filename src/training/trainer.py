"""GRPO训练器"""

import copy
import json
import os
import random
from datetime import datetime

import torch

from ..grpo.algorithm import generate_rollout_data, grpo_loss
from ..utils.device import get_device


class GRPOTrainer:
    """GRPO训练器类"""

    def __init__(self, model, tokenizer, config):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            tokenizer: 分词器
            config: 配置字典
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = get_device()

        # 创建实验目录
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"output/experiments/exp_{self.experiment_id}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/metrics", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/samples", exist_ok=True)

        # 初始化指标记录
        self.metrics_history = []
        self.step_count = 0

        # 保存实验配置
        self._save_experiment_config()

    def _save_experiment_config(self):
        """保存实验配置"""
        config_data = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
        }

        with open(f"{self.experiment_dir}/experiment_config.json", "w") as f:
            json.dump(config_data, f, indent=2, default=str)

    def _calculate_output_stats(self, rollout_data):
        """计算输出文本统计信息"""
        completions = rollout_data["formatted_completions"]

        lengths = []
        for completion in completions:
            text = completion[0]["content"]
            lengths.append(len(text.split()))  # 词数

        return {
            "avg_output_length": sum(lengths) / len(lengths),
            "min_output_length": min(lengths),
            "max_output_length": max(lengths),
        }

    def _save_training_metrics(
        self, iteration, step, grpo_iter, loss, avg_reward, rollout_data
    ):
        """保存训练指标"""
        self.step_count += 1

        # 计算输出统计
        output_stats = self._calculate_output_stats(rollout_data)

        metrics = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "step_count": self.step_count,
            "iteration": iteration,
            "step": step,
            "grpo_iter": grpo_iter,
            "loss": float(loss),
            "avg_reward": float(avg_reward),
            "output_stats": output_stats,
        }

        self.metrics_history.append(metrics)

        # 保存到文件
        with open(f"{self.experiment_dir}/metrics/training_metrics.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def _save_checkpoint(self, iteration, step):
        """保存模型检查点"""
        checkpoint_dir = (
            f"{self.experiment_dir}/checkpoints/iteration_{iteration}_step_{step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        print(f"Checkpoint saved: {checkpoint_dir}")

    def optimize_model_memory(self):
        """
        Optimizes the model to use less memory during training.

        Returns:
            The optimized model.

        Explanation:
            1. Sets the model to training mode.
            2. Disables KV caching to save memory.
            3. Enables gradient checkpointing to trade computation for memory.
            4. Returns the optimized model ready for memory-efficient training.
        """
        self.model.train()
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        return self.model

    def train(self, train_data, reward_function):
        """
        Train the model using GRPO

        Args:
            train_data: 训练数据
            reward_function: 奖励函数

        Returns:
            The trained model.
        """
        # 优化模型内存使用
        self.model = self.optimize_model_memory()

        # 训练配置
        num_iterations = self.config.get("num_iterations", 1)
        num_steps = self.config.get("num_steps", 500)
        batch_size = self.config.get("batch_size", 4)
        num_generations = self.config.get("num_generations", 4)
        max_completion_length = self.config.get("max_completion_length", 128)
        beta = self.config.get("beta", 0.1)
        learning_rate = self.config.get("learning_rate", 5e-6)
        mu = self.config.get("mu", 3)
        epsilon = self.config.get("epsilon", 0.2)
        grad_clip_norm = self.config.get("grad_clip_norm", 0.1)

        # 外层循环：迭代GRPO更新
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration+1}/{num_iterations}")

            # 创建参考模型（深拷贝）并设置为评估模式
            ref_model = copy.deepcopy(self.model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            print("Reference model created.")

            # 重新初始化优化器
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            self.model.train()

            # 内层循环：训练步骤
            for step in range(num_steps):
                batch_samples = random.sample(train_data, batch_size)

                with torch.no_grad():
                    rollout_data = generate_rollout_data(
                        self.model,
                        ref_model,
                        self.tokenizer,
                        batch_samples,
                        num_generations,
                        max_completion_length,
                    )

                for grpo_iter in range(mu):
                    loss, avg_reward = grpo_loss(
                        self.model,
                        ref_model,
                        rollout_data,
                        self.tokenizer,
                        reward_function,
                        beta=beta,
                        epsilon=epsilon,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=grad_clip_norm
                    )
                    optimizer.step()

                    print(
                        f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                        f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item():.4f}, avg_reward: {avg_reward:.4f}"
                    )

                    # 保存训练指标
                    self._save_training_metrics(
                        iteration + 1,
                        step + 1,
                        grpo_iter + 1,
                        loss.item(),
                        avg_reward,
                        rollout_data,
                    )

                # 保存检查点（每50步）
                if (step + 1) % 50 == 0:
                    self._save_checkpoint(iteration + 1, step + 1)

        return self.model
