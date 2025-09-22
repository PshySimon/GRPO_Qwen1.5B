#!/bin/bash

# VERL 8*V100 3B模型 GRPO训练启动脚本

echo "🚀 VERL GRPO 多卡训练启动脚本"
echo "目标: 8*V100 + 3B模型 + GRPO算法"

# 检查GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $NUM_GPUS 个GPU"

if [ $NUM_GPUS -eq 0 ]; then
    echo "❌ 错误: 未检测到GPU"
    exit 1
fi

if [ $NUM_GPUS -lt 8 ]; then
    echo "⚠️  警告: 检测到${NUM_GPUS}个GPU，建议使用8个GPU获得最佳性能"
    echo "是否继续？(y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "训练取消"
        exit 1
    fi
fi

# 设置分布式训练环境变量
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"  # 使用不同端口避免冲突
export WORLD_SIZE=$NUM_GPUS
export NCCL_DEBUG=INFO      # 调试NCCL通信

# V100特定优化
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1    # 如果没有InfiniBand

echo "📊 训练配置:"
echo "  - 算法: GRPO (Group Relative Policy Optimization)"
echo "  - 模型: 3B参数"
echo "  - GPU: ${NUM_GPUS}*V100"
echo "  - 分布式: FSDP2"
echo "  - 精度: FP16"
echo "  - Master地址: $MASTER_ADDR:$MASTER_PORT"
echo "  - 世界大小: $WORLD_SIZE"

# 启动多进程GRPO训练
echo "🎯 启动VERL GRPO训练..."
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    scripts/train_verl.py \
    "$@"

if [ $? -eq 0 ]; then
    echo "✅ GRPO训练完成！"
else
    echo "❌ 训练失败，请检查日志"
fi
