"""使用VERL框架的多卡GRPO训练脚本"""

import argparse
import os
from datetime import datetime

import torch
import yaml
from src.data.dataset import prepare_dataset, split_dataset
from src.rewards.rewards import combined_reward
from src.utils.config import load_all_configs
from src.utils.logging import setup_logger
from src.utils.random import set_random_seed
from transformers import AutoTokenizer

# VERL imports
from verl import DataPool, RewardFunction
from verl.models import AutoModelForCausalLMWithValueHead
from verl.trainer import GRPOTrainer  # 使用GRPO而不是PPO
from verl.utils import init_distributed, setup_fsdp


class GSM8KRewardFunction(RewardFunction):
    """GSM8K数学问题奖励函数"""

    def __init__(self):
        super().__init__()

    def __call__(self, prompts, responses, **kwargs):
        """计算奖励"""
        # 转换为我们的奖励函数格式
        completions = [[{"content": response}] for response in responses]
        answers = kwargs.get("answers", [])

        return combined_reward(prompts, completions, answers)


def prepare_verl_dataset(train_data, eval_data):
    """准备VERL格式的数据集"""

    # 训练数据
    train_prompts = [item["prompt"] for item in train_data]
    train_answers = [item["answer"] for item in train_data]

    # 评估数据
    eval_prompts = [item["prompt"] for item in eval_data]
    eval_answers = [item["answer"] for item in eval_data]

    # 创建VERL DataPool
    train_pool = DataPool(prompts=train_prompts, metadata={"answers": train_answers})

    eval_pool = DataPool(prompts=eval_prompts, metadata={"answers": eval_answers})

    return train_pool, eval_pool


def create_verl_trainer(model, tokenizer, verl_config):
    """创建VERL GRPO训练器"""

    # 设置FSDP（V100不支持DeepSpeed，使用FSDP）
    model = setup_fsdp(
        model,
        strategy="fsdp2",
        cpu_offload=verl_config["optimization"]["fsdp_cpu_offload"],
        mixed_precision="fp16" if verl_config["optimization"]["fp16"] else None,
    )

    # 包装模型为带价值头的模型
    model_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        num_value_heads=1,
    )

    # 创建奖励函数
    reward_fn = GSM8KRewardFunction()

    # GRPO训练配置（针对3B模型和V100优化）
    grpo_config = {
        "algorithm": "grpo",  # 指定GRPO算法
        "learning_rate": float(verl_config["grpo"]["learning_rate"]),
        "batch_size": int(verl_config["grpo"]["batch_size"]),
        "mini_batch_size": int(verl_config["grpo"]["mini_batch_size"]),
        "num_epochs": int(verl_config["grpo"]["num_epochs"]),
        "max_grad_norm": float(verl_config["grpo"]["max_grad_norm"]),
        "beta": float(verl_config["grpo"]["beta"]),
        "epsilon": float(verl_config["grpo"]["epsilon"]),
        "generation_kwargs": {
            "max_new_tokens": int(verl_config["generation"]["max_new_tokens"]),
            "num_return_sequences": int(
                verl_config["generation"]["num_return_sequences"]
            ),
            "do_sample": verl_config["generation"]["do_sample"],
            "temperature": float(verl_config["generation"]["temperature"]),
            "top_p": float(verl_config["generation"]["top_p"]),
            "top_k": int(verl_config["generation"]["top_k"]),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        # V100优化配置
        "gradient_checkpointing": verl_config["optimization"]["gradient_checkpointing"],
        "use_cache": verl_config["optimization"]["use_cache"],
        "fp16": verl_config["optimization"]["fp16"],
    }

    # 创建GRPO训练器
    trainer = GRPOTrainer(
        model=model_with_value,
        tokenizer=tokenizer,
        reward_function=reward_fn,
        **grpo_config,
    )

    return trainer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VERL多卡GRPO训练脚本")
    parser.add_argument("--resume", action="store_true", help="从断点续训")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地GPU排名")

    args = parser.parse_args()

    # 初始化分布式训练
    init_distributed()

    # 设置随机种子
    set_random_seed(42)

    # 检查是否续训
    resume_config = None
    if args.resume:
        resume_file = "config/resume.yaml"
        if os.path.exists(resume_file):
            with open(resume_file, "r") as f:
                resume_config = yaml.safe_load(f)
            print(
                f"发现断点续训配置，将从实验 {resume_config['experiment_id']} 继续训练"
            )
        else:
            print("未找到断点续训配置，将从头开始训练")

    # 加载配置
    configs = load_all_configs()
    verl_config = configs["verl_config"]  # 使用VERL专用配置
    model_config = configs["model"]
    logging_config = configs["logging"]

    # 设置日志
    logger = setup_logger(logging_config["log_dir"], logging_config["level"])

    # 创建实验目录
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"output/experiments/exp_verl_{experiment_id}"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{experiment_dir}/metrics", exist_ok=True)

    logger.info(f"VERL多卡训练开始，实验ID: {experiment_id}")
    logger.info(f"实验目录: {experiment_dir}")

    # 加载模型和tokenizer
    logger.info("加载模型...")
    model_name = model_config["model_name"]

    # 对于VERL，我们只需要加载基础模型
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # VERL会处理多卡分布
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # 设置tokenizer配置
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    logger.info("模型加载完成")

    # 准备数据
    logger.info("准备数据集...")
    all_data = prepare_dataset("train")
    train_data, eval_data = split_dataset(all_data, eval_size=50)

    logger.info(f"Training data size: {len(train_data)}")
    logger.info(f"Evaluation data size: {len(eval_data)}")

    # 准备VERL数据集
    train_pool, eval_pool = prepare_verl_dataset(train_data, eval_data)

    # 创建VERL训练器
    logger.info("创建VERL GRPO训练器...")
    trainer = create_verl_trainer(model, tokenizer, verl_config)

    # 训练配置
    num_iterations = int(verl_config["training"]["num_iterations"])
    num_steps = int(verl_config["training"]["num_steps"])
    save_interval = int(verl_config["training"]["save_interval"])
    eval_interval = int(verl_config["training"]["eval_interval"])
    log_interval = int(verl_config["training"]["log_interval"])

    logger.info(
        f"开始VERL训练: {num_iterations} iterations, {num_steps} steps per iteration"
    )

    # 训练循环
    for iteration in range(num_iterations):
        logger.info(f"\n=== Iteration {iteration+1}/{num_iterations} ===")

        # VERL GRPO训练步骤
        for step in range(num_steps):
            # 从数据池采样
            batch = train_pool.sample(verl_config["grpo"]["batch_size"])

            # GRPO训练一步
            metrics = trainer.train_step(batch)

            # 记录指标
            if step % log_interval == 0:
                logger.info(f"Step {step+1}/{num_steps}: {metrics}")

            # 保存检查点
            if (step + 1) % save_interval == 0:
                checkpoint_dir = f"{experiment_dir}/checkpoints/iteration_{iteration+1}_step_{step+1}"
                trainer.save_checkpoint(checkpoint_dir)
                logger.info(f"Checkpoint saved: {checkpoint_dir}")

            # 定期评估
            if (step + 1) % eval_interval == 0 and eval_pool:
                logger.info("中间评估...")
                eval_metrics = trainer.evaluate(eval_pool)
                logger.info(f"Evaluation metrics: {eval_metrics}")

        # 每个iteration后完整评估
        if eval_pool:
            logger.info("Iteration结束评估...")
            eval_metrics = trainer.evaluate(eval_pool)
            logger.info(f"Final evaluation metrics: {eval_metrics}")

    # 保存最终模型
    final_model_dir = f"{experiment_dir}/final_model"
    trainer.save_model(final_model_dir)
    logger.info(f"最终模型保存在: {final_model_dir}")

    # 保存训练状态
    resume_config = {
        "experiment_id": f"verl_{experiment_id}",
        "experiment_dir": experiment_dir,
        "training_state": {
            "current_iteration": num_iterations,
            "current_step": num_steps,
            "total_steps_completed": num_iterations * num_steps,
            "best_reward": 0.0,
            "experiment_dir": experiment_dir,
        },
        "config": verl_config,
        "last_saved": datetime.now().isoformat(),
    }

    with open("config/resume.yaml", "w") as f:
        yaml.dump(resume_config, f, indent=2, default_flow_style=False)

    logger.info("训练完成！")
    logger.info(f"实验数据保存在: {experiment_dir}")


if __name__ == "__main__":
    main()
