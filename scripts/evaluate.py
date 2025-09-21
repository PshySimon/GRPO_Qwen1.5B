"""简洁版评估脚本"""

import argparse
import os
from pathlib import Path

import torch
from src.data.dataset import prepare_dataset
from src.data.utils import (
    extract_answer_from_model_output,
    extract_last_number,
    extract_single_number,
)
from src.utils.device import get_device
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_checkpoint_step(checkpoint_name):
    """从checkpoint名称中提取step数字用于排序"""
    import re

    match = re.search(r"iteration_(\d+)_step_(\d+)", checkpoint_name)
    if match:
        iteration = int(match.group(1))
        step = int(match.group(2))
        return iteration, step
    return 0, 0


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

    latest = max(experiment_dirs, key=lambda x: x.name)
    return str(latest)


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
            latest_checkpoint = max(
                checkpoints, key=lambda x: parse_checkpoint_step(x.name)
            )
            return str(latest_checkpoint)

    return None


def evaluate_model(model, tokenizer, eval_examples, device, verbose=False):
    """评估模型"""
    model.eval()
    correct = 0
    total = len(eval_examples)

    if verbose:
        print(f"开始评估 {total} 个样本（详细模式）")
        eval_iterator = eval_examples
    else:
        eval_iterator = tqdm(eval_examples, desc="评估进度", unit="样本")

    for example in eval_iterator:
        full_prompt = example["prompt"]
        expected = example["answer"]

        # 推理
        inputs = tokenizer.encode(
            full_prompt, return_tensors="pt", padding=True, padding_side="left"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
                do_sample=False,  # 确定性生成
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            # 提取答案并检查正确性
            predicted = extract_answer_from_model_output(response)

            # 尝试不同的匹配方法
            if predicted == expected:  # 精确匹配
                is_correct = True
            else:
                # 尝试数字匹配
                pred_num = extract_single_number(str(predicted))
                exp_num = extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # 尝试最后一个数字匹配
                    pred_num = extract_last_number(str(predicted))
                    exp_num = extract_last_number(str(expected))
                    is_correct = (
                        pred_num is not None
                        and exp_num is not None
                        and pred_num == exp_num
                    )

            if is_correct:
                correct += 1

            # 详细输出
            if verbose:
                print(f"\n问题: {full_prompt[:100]}...")
                print(f"期望答案: {expected}")
                print(f"提取答案: {predicted}")
                print(f"正确: {'✓' if is_correct else '✗'}")
                print("-" * 50)

        except Exception as e:
            if verbose:
                print(f"解析失败: {e}")

    # 计算并显示最终准确率
    accuracy = (correct / total) * 100
    print(f"\n准确率: {accuracy:.2f}% ({correct}/{total})")

    model.train()
    return accuracy


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估GRPO模型")
    parser.add_argument("--checkpoint", type=str, help="指定checkpoint路径")
    parser.add_argument("--experiment", type=str, help="指定实验目录")
    parser.add_argument("--baseline", action="store_true", help="评估原始基准模型")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    parser.add_argument(
        "--num_samples", type=int, default=50, help="评估样本数量（默认50）"
    )

    args = parser.parse_args()
    device = get_device()

    # 确定模型路径
    if args.baseline:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"评估原始基准模型: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    elif args.checkpoint:
        model_path = args.checkpoint
        print(f"评估指定checkpoint: {model_path}")
        if not os.path.exists(model_path):
            print(f"错误: Checkpoint路径不存在: {model_path}")
            return

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    elif args.experiment:
        model_path = find_model_path(args.experiment)
        if not model_path:
            print(f"错误: 在实验中未找到模型: {args.experiment}")
            return

        print(f"评估实验模型: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    else:
        # 自动找最新实验
        latest_exp = find_latest_experiment()
        if latest_exp:
            model_path = find_model_path(latest_exp)
            if model_path:
                print(f"评估最新实验模型: {model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                ).to(device)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, padding_side="left"
                )
            else:
                print(f"错误: 在最新实验中未找到模型: {latest_exp}")
                return
        else:
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            print(f"未找到实验，评估原始模型: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # 设置tokenizer配置
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 准备评估数据
    all_data = prepare_dataset("train")
    eval_data = all_data[: args.num_samples]

    # 评估模型（不保存返回值，避免警告）
    evaluate_model(model, tokenizer, eval_data, device, verbose=args.verbose)


if __name__ == "__main__":
    main()
