"""训练脚本"""

import torch
from src.data.dataset import prepare_dataset, split_dataset
from src.rewards.rewards import combined_reward
from src.training.trainer import GRPOTrainer
from src.utils.config import load_all_configs
from src.utils.logging import setup_logger
from src.utils.random import set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # 设置随机种子
    set_random_seed(42)

    # 加载所有配置
    configs = load_all_configs()
    training_config = configs["training"]
    model_config = configs["model"]
    logging_config = configs["logging"]

    # 设置日志
    logger = setup_logger(logging_config["log_dir"], logging_config["level"])

    logger.info("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"],
        attn_implementation=model_config["attn_implementation"],
        torch_dtype=getattr(torch, model_config["torch_dtype"]),
        device_map=model_config["device_map"],
    )
    logger.info("Model downloaded")

    logger.info("model config:")
    logger.info(model.config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"], padding_side="left"
    )

    # 设置tokenizer配置
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    logger.info("model new config:")
    logger.info(model.config)

    # 准备数据
    all_data = prepare_dataset("train")
    train_data, eval_data = split_dataset(all_data, eval_size=50)

    logger.info(f"Training data size: {len(train_data)}")
    logger.info(f"Evaluation data size: {len(eval_data)}")

    # 创建训练器
    trainer = GRPOTrainer(model, tokenizer, training_config)

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
