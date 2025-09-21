#!/bin/bash

# GRPO项目下载脚本
set -e

echo "开始下载GRPO项目资源..."

# 安装modelscope命令行工具
pip install modelscope

# 创建目录
mkdir -p ./models

# 下载模型
echo "下载Qwen2.5-1.5B模型..."
modelscope download --model qwen/Qwen2.5-1.5B-Instruct --local_dir ./models/qwen2.5-1.5b-instruct

echo "下载完成！"
echo "模型位置: ./models/qwen2.5-1.5b/"
