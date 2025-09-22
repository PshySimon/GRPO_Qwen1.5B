"""测试微调后的模型 - 支持指定checkpoint"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from src.data.utils import SYSTEM_PROMPT, build_prompt, extract_answer_from_model_output
from src.utils.device import get_device
from src.utils.logging import setup_logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_latest_experiment():
    """找到最新的实验目录"""
    experiments_dir = Path("output/experiments")
    if not experiments_dir.exists():
        return None

    experiment_dirs = [
        d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")
    ]
    if not experiment_dirs:
        return None

    # 按时间戳排序，返回最新的
    latest = max(experiment_dirs, key=lambda x: x.name)
    return str(latest)


def parse_checkpoint_step(checkpoint_name):
    """从checkpoint名称中提取step数字用于排序"""
    import re

    # 匹配 iteration_X_step_Y 格式
    match = re.search(r"iteration_(\d+)_step_(\d+)", checkpoint_name)
    if match:
        iteration = int(match.group(1))
        step = int(match.group(2))
        return iteration, step
    return 0, 0


def find_model_path(experiment_dir):
    """在实验目录中找到模型路径"""
    experiment_path = Path(experiment_dir)

    # 优先使用final_model
    final_model_path = experiment_path / "final_model"
    if final_model_path.exists():
        return str(final_model_path)

    # 如果没有final_model，使用最新的checkpoint
    checkpoints_dir = experiment_path / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
        if checkpoints:
            # 按iteration和step数字排序，找最新的
            latest_checkpoint = max(
                checkpoints, key=lambda x: parse_checkpoint_step(x.name)
            )
            return str(latest_checkpoint)

    return None


def list_checkpoints(experiment_dir):
    """列出实验目录中的所有checkpoint"""
    experiment_path = Path(experiment_dir)
    checkpoints_dir = experiment_path / "checkpoints"

    if not checkpoints_dir.exists():
        return []

    checkpoints = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
    # 按iteration和step数字排序
    sorted_checkpoints = sorted(
        checkpoints, key=lambda x: parse_checkpoint_step(x.name)
    )
    return [str(d) for d in sorted_checkpoints]


def load_resume_config():
    """加载断点续训配置"""
    resume_file = "config/resume.yaml"
    if os.path.exists(resume_file):
        with open(resume_file, "r") as f:
            return yaml.safe_load(f)
    return None


def test_model_with_path(model_path, logger):
    """使用指定路径测试模型"""
    device = get_device()

    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return False

    # 加载模型
    try:
        logger.info(f"正在加载模型: {model_path}")
        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)

        loaded_tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left"
        )
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False

    logger.info("\n🤖 GRPO数学问题求解器 - 交互式对话模式")
    logger.info("=" * 60)
    logger.info("输入数学问题，模型会为你求解！")
    logger.info("输入 'quit', 'exit', 'q' 退出程序")
    logger.info("=" * 60)

    while True:
        try:
            # 获取用户输入
            user_input = input("\n🧮 请输入数学问题: ").strip()

            # 检查退出命令
            if user_input.lower() in ["quit", "exit", "q", ""]:
                print("👋 再见！")
                break

            if not user_input:
                continue

            print(f"\n📝 问题: {user_input}")
            print("🤖 回答: ", end="", flush=True)

            # 准备提示
            test_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ]
            test_prompt = build_prompt(test_messages)

            # 生成响应 - 流式输出
            test_input_ids = loaded_tokenizer.encode(
                test_prompt, return_tensors="pt", padding=True, padding_side="left"
            ).to(device)

            # 使用标准generate方法（确保正确性）
            with torch.no_grad():
                test_output_ids = loaded_model.generate(
                    test_input_ids,
                    max_new_tokens=400,
                    num_return_sequences=1,
                    pad_token_id=loaded_tokenizer.pad_token_id,
                    eos_token_id=loaded_tokenizer.eos_token_id,
                    do_sample=False,  # 确定性生成
                    early_stopping=False,
                    # 覆盖generation_config中的采样参数
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                )

            # 获取完整响应
            test_response = loaded_tokenizer.decode(
                test_output_ids[0], skip_special_tokens=True
            )

            # 提取新生成的部分（去掉prompt）
            original_prompt = loaded_tokenizer.decode(
                test_input_ids[0], skip_special_tokens=True
            )
            if test_response.startswith(original_prompt):
                generated_part = test_response[len(original_prompt) :].strip()
            else:
                generated_part = test_response

            # 流式显示效果（模拟）
            import time

            for char in generated_part:
                print(char, end="", flush=True)
                time.sleep(0.01)  # 打字机效果

            print()  # 换行

            # 提取最终答案
            try:
                extracted_answer = extract_answer_from_model_output(generated_part)
                if extracted_answer:
                    print(f"\n💡 提取的答案: {extracted_answer}")
                else:
                    print("\n⚠️  无法从回答中提取明确答案")

            except Exception as e:
                print(f"\n❌ 答案提取失败: {e}")

        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("请重试或输入 'quit' 退出")

    logger.info("\n测试会话结束！")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试微调后的GRPO模型")
    parser.add_argument("--checkpoint", type=str, help="指定checkpoint路径")
    parser.add_argument("--experiment", type=str, help="指定实验目录")
    parser.add_argument("--list", action="store_true", help="列出可用的checkpoint")

    args = parser.parse_args()

    logger = setup_logger()

    # 如果指定了checkpoint路径，直接测试
    if args.checkpoint:
        logger.info(f"使用指定的checkpoint: {args.checkpoint}")
        test_model_with_path(args.checkpoint, logger)
        return

    # 如果指定了实验目录
    if args.experiment:
        if args.list:
            # 列出该实验的所有checkpoint
            checkpoints = list_checkpoints(args.experiment)
            if checkpoints:
                logger.info(f"实验 {args.experiment} 中的checkpoint:")
                for i, cp in enumerate(checkpoints, 1):
                    logger.info(f"  {i}. {cp}")
                logger.info(
                    "\n使用方法: python scripts/test_model.py --checkpoint <路径>"
                )
            else:
                logger.info(f"实验 {args.experiment} 中没有找到checkpoint")
            return
        else:
            # 使用该实验的最佳模型
            model_path = find_model_path(args.experiment)
            if model_path:
                logger.info(f"使用实验 {args.experiment} 的模型: {model_path}")
                test_model_with_path(model_path, logger)
            else:
                logger.error(f"在实验 {args.experiment} 中未找到模型")
            return

    # 如果只是列出，找最新实验
    if args.list:
        latest_exp = find_latest_experiment()
        if latest_exp:
            logger.info(f"最新实验: {latest_exp}")
            checkpoints = list_checkpoints(latest_exp)
            if checkpoints:
                logger.info("可用的checkpoint:")
                for i, cp in enumerate(checkpoints, 1):
                    logger.info(f"  {i}. {cp}")
                logger.info(
                    "\n使用方法: python scripts/test_model.py --checkpoint <路径>"
                )
            else:
                logger.info("没有找到checkpoint")
        else:
            logger.info("没有找到任何实验")
        return

    # 默认行为：自动找模型测试
    model_path = None

    # 尝试从resume配置获取
    resume_config = load_resume_config()
    if resume_config:
        experiment_dir = resume_config["experiment_dir"]
        model_path = find_model_path(experiment_dir)
        if model_path:
            logger.info(f"从resume配置找到模型: {model_path}")
        else:
            logger.error(f"在实验目录中未找到模型: {experiment_dir}")
            return
    else:
        # 尝试找到最新实验
        latest_exp = find_latest_experiment()
        if latest_exp:
            model_path = find_model_path(latest_exp)
            if model_path:
                logger.info(f"找到最新实验模型: {model_path}")
            else:
                logger.error(f"在最新实验中未找到模型: {latest_exp}")
                return
        else:
            # 回退到旧路径
            model_path = "grpo_finetuned_model"
            if not os.path.exists(model_path):
                logger.error("未找到任何微调模型！")
                logger.info("使用 --list 查看可用的checkpoint")
                return
            logger.info(f"使用旧路径模型: {model_path}")

    if model_path:
        test_model_with_path(model_path, logger)


if __name__ == "__main__":
    main()
