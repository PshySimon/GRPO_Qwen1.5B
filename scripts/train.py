"""训练脚本"""

import argparse
import os

import torch
import yaml
from src.data.dataset import prepare_dataset, split_dataset
from src.rewards.rewards import combined_reward
from src.training.trainer import GRPOTrainer
from src.utils.config import load_all_configs
from src.utils.logging import setup_logger
from src.utils.random import set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_resume_config():
    """加载断点续训配置"""
    resume_file = "config/resume.yaml"
    if os.path.exists(resume_file):
        with open(resume_file, "r") as f:
            return yaml.safe_load(f)
    return None


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GRPO训练脚本")
    parser.add_argument("--resume", action="store_true", help="从断点续训")
    args = parser.parse_args()

    # 设置随机种子
    set_random_seed(42)

    # 检查是否续训
    resume_config = None
    if args.resume:
        resume_config = load_resume_config()
        if resume_config:
            print(
                f"发现断点续训配置，将从实验 {resume_config['experiment_id']} 继续训练"
            )
        else:
            print("未找到断点续训配置，将从头开始训练")

    # 加载配置
    if resume_config:
        # 使用保存的配置
        training_config = resume_config["config"]
        configs = load_all_configs()
        model_config = configs["model"]
        logging_config = configs["logging"]
        print("使用断点续训配置")
    else:
        # 使用默认配置
        configs = load_all_configs()
        training_config = configs["training"]
        model_config = configs["model"]
        logging_config = configs["logging"]

    # 设置日志
    logger = setup_logger(logging_config["log_dir"], logging_config["level"])

    # 加载模型
    if resume_config:
        # 从检查点加载模型
        checkpoint_dir = resume_config["experiment_dir"] + "/checkpoints"
        if os.path.exists(checkpoint_dir):
            # 找到最新的检查点
            checkpoints = [
                d for d in os.listdir(checkpoint_dir) if d.startswith("iteration_")
            ]
            if checkpoints:
                latest_checkpoint = max(checkpoints)
                model_path = os.path.join(checkpoint_dir, latest_checkpoint)
                logger.info(f"从检查点加载模型: {model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, padding_side="left"
                )
            else:
                logger.info("未找到检查点，从原始模型开始")
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["model_name"],
                    torch_dtype=torch.float16,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config["model_name"], padding_side="left"
                )
        else:
            logger.info("检查点目录不存在，从原始模型开始")
            model = AutoModelForCausalLM.from_pretrained(
                model_config["model_name"],
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_config["model_name"], padding_side="left"
            )
    else:
        logger.info("加载原始模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name"],
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_name"], padding_side="left"
        )

    # 移动模型到设备
    from src.utils.device import get_device

    device = get_device()
    model = model.to(device)

    # 设置tokenizer配置
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    logger.info("模型加载完成")

    # 准备数据
    all_data = prepare_dataset("train")
    train_data, eval_data = split_dataset(all_data, eval_size=50)

    logger.info(f"Training data size: {len(train_data)}")
    logger.info(f"Evaluation data size: {len(eval_data)}")

    # 创建训练器
    trainer = GRPOTrainer(model, tokenizer, training_config, resume_config)

    logger.info("\nStarting RL fine-tuning using GRPO...")
    model = trainer.train(train_data, reward_function=combined_reward)

    logger.info("Training completed.")
    logger.info(f"实验数据保存在: {trainer.experiment_dir}")

    # 保存最终模型到实验目录
    final_model_dir = f"{trainer.experiment_dir}/final_model"
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    logger.info(f"最终模型保存在: {final_model_dir}")
    logger.info("使用以下命令查看训练图表:")
    logger.info(f"python scripts/visualize_experiment.py {trainer.experiment_dir}")


if __name__ == "__main__":
    main()
