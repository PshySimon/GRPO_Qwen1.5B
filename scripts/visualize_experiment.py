#!/usr/bin/env python3
"""实验数据可视化脚本"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_experiment_data(experiment_dir):
    """加载实验数据"""
    metrics_file = Path(experiment_dir) / "metrics" / "training_metrics.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"指标文件不存在: {metrics_file}")

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    return metrics


def plot_loss(metrics, save_path):
    """绘制loss曲线"""
    steps = [m["step_count"] for m in metrics]
    losses = [m["loss"] for m in metrics]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, "b-", linewidth=1)
    plt.title("Training Loss Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss图表已保存: {save_path}")


def plot_reward(metrics, save_path):
    """绘制reward曲线"""
    steps = [m["step_count"] for m in metrics]

    # 检查是否有奖励分解数据
    has_breakdown = any("reward_breakdown" in m for m in metrics)

    if has_breakdown:
        # 分别绘制correctness和format reward
        correctness_rewards = []
        format_rewards = []
        combined_rewards = []

        for m in metrics:
            if "reward_breakdown" in m:
                correctness_rewards.append(
                    m["reward_breakdown"]["avg_correctness_reward"]
                )
                format_rewards.append(m["reward_breakdown"]["avg_format_reward"])
                combined_rewards.append(m["reward_breakdown"]["avg_combined_reward"])
            else:
                # 兼容旧数据
                correctness_rewards.append(0)
                format_rewards.append(0)
                combined_rewards.append(m["avg_reward"])

        plt.figure(figsize=(12, 8))
        plt.plot(steps, combined_rewards, "g-", linewidth=2, label="Combined Reward")
        plt.plot(
            steps, correctness_rewards, "b-", linewidth=1.5, label="Correctness Reward"
        )
        plt.plot(steps, format_rewards, "r-", linewidth=1.5, label="Format Reward")
        plt.title("Reward Breakdown Over Time")
        plt.xlabel("Training Steps")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Reward分解图表已保存: {save_path}")
    else:
        # 兼容旧版本，只有总奖励
        rewards = [m["avg_reward"] for m in metrics]
        plt.figure(figsize=(12, 6))
        plt.plot(steps, rewards, "g-", linewidth=1)
        plt.title("Average Reward Over Time")
        plt.xlabel("Training Steps")
        plt.ylabel("Average Reward")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Reward图表已保存: {save_path}")


def plot_output_length(metrics, save_path):
    """绘制输出长度曲线"""
    steps = [m["step_count"] for m in metrics]
    avg_lengths = [m["output_stats"]["avg_output_length"] for m in metrics]
    min_lengths = [m["output_stats"]["min_output_length"] for m in metrics]
    max_lengths = [m["output_stats"]["max_output_length"] for m in metrics]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, avg_lengths, "r-", linewidth=1, label="Average Length")
    plt.plot(steps, min_lengths, "r--", alpha=0.7, label="Min Length")
    plt.plot(steps, max_lengths, "r--", alpha=0.7, label="Max Length")
    plt.fill_between(steps, min_lengths, max_lengths, alpha=0.2, color="red")

    plt.title("Output Text Length Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Text Length (words)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"输出长度图表已保存: {save_path}")


def plot_loss_vs_reward(metrics, save_path):
    """绘制loss vs reward散点图"""
    losses = [m["loss"] for m in metrics]
    rewards = [m["avg_reward"] for m in metrics]

    plt.figure(figsize=(10, 8))
    plt.scatter(losses, rewards, alpha=0.6, c=range(len(losses)), cmap="viridis")
    plt.colorbar(label="Training Steps")
    plt.title("Loss vs Average Reward")
    plt.xlabel("Loss")
    plt.ylabel("Average Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss vs Reward图表已保存: {save_path}")


def plot_training_progress(metrics, save_path):
    """绘制训练进度综合图"""
    has_breakdown = any("reward_breakdown" in m for m in metrics)

    if has_breakdown:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    steps = [m["step_count"] for m in metrics]

    # Loss
    losses = [m["loss"] for m in metrics]
    ax1.plot(steps, losses, "b-", linewidth=1)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Reward (with breakdown if available)
    if has_breakdown:
        correctness_rewards = []
        format_rewards = []
        combined_rewards = []

        for m in metrics:
            if "reward_breakdown" in m:
                correctness_rewards.append(
                    m["reward_breakdown"]["avg_correctness_reward"]
                )
                format_rewards.append(m["reward_breakdown"]["avg_format_reward"])
                combined_rewards.append(m["reward_breakdown"]["avg_combined_reward"])
            else:
                correctness_rewards.append(0)
                format_rewards.append(0)
                combined_rewards.append(m["avg_reward"])

        ax2.plot(steps, combined_rewards, "g-", linewidth=2, label="Combined")
        ax2.plot(steps, correctness_rewards, "b-", linewidth=1, label="Correctness")
        ax2.plot(steps, format_rewards, "r-", linewidth=1, label="Format")
        ax2.set_title("Reward Breakdown")
        ax2.legend()
    else:
        rewards = [m["avg_reward"] for m in metrics]
        ax2.plot(steps, rewards, "g-", linewidth=1)
        ax2.set_title("Average Reward")

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)

    # Output Length
    avg_lengths = [m["output_stats"]["avg_output_length"] for m in metrics]
    ax3.plot(steps, avg_lengths, "r-", linewidth=1)
    ax3.set_title("Average Output Length")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Length (words)")
    ax3.grid(True, alpha=0.3)

    # Loss vs Reward
    if has_breakdown:
        combined_rewards_for_scatter = combined_rewards
    else:
        combined_rewards_for_scatter = [m["avg_reward"] for m in metrics]

    ax4.scatter(
        losses, combined_rewards_for_scatter, alpha=0.6, c=steps, cmap="viridis"
    )
    ax4.set_title("Loss vs Combined Reward")
    ax4.set_xlabel("Loss")
    ax4.set_ylabel("Combined Reward")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练进度综合图已保存: {save_path}")


def generate_summary_stats(metrics):
    """生成统计摘要"""
    losses = [m["loss"] for m in metrics]
    rewards = [m["avg_reward"] for m in metrics]
    avg_lengths = [m["output_stats"]["avg_output_length"] for m in metrics]

    summary = {
        "total_steps": len(metrics),
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "final_reward": rewards[-1],
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "avg_output_length": np.mean(avg_lengths),
        "min_output_length": min(avg_lengths),
        "max_output_length": max(avg_lengths),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="可视化GRPO实验数据")
    parser.add_argument("experiment_dir", help="实验目录路径")
    parser.add_argument("--output_dir", default="output/plots", help="图表输出目录")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # 加载数据
        print(f"加载实验数据: {args.experiment_dir}")
        metrics = load_experiment_data(args.experiment_dir)

        # 生成图表
        print("生成可视化图表...")
        plot_loss(metrics, f"{args.output_dir}/loss.png")
        plot_reward(metrics, f"{args.output_dir}/reward.png")
        plot_output_length(metrics, f"{args.output_dir}/output_length.png")
        plot_loss_vs_reward(metrics, f"{args.output_dir}/loss_vs_reward.png")
        plot_training_progress(metrics, f"{args.output_dir}/training_progress.png")

        # 生成统计摘要
        summary = generate_summary_stats(metrics)
        with open(f"{args.output_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n=== 训练统计摘要 ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

        print(f"\n所有图表已保存到: {args.output_dir}")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
