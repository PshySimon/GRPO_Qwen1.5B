# GRPO Qwen2.5-1.5B 项目

基于 GRPO (Group Relative Policy Optimization) 算法微调 Qwen2.5-1.5B-Instruct 模型进行数学问题求解的项目。

## ✨ 主要特性

- 🎯 **完整的GRPO实现**: 从头实现Group Relative Policy Optimization算法
- 📊 **详细指标记录**: 记录loss、reward、输出长度等训练指标
- 📈 **可视化分析**: 自动生成训练过程图表和统计报告
- 🔄 **断点续训**: 支持训练中断后无缝续训
- 🎨 **奖励分解**: 分别显示正确性奖励和格式奖励
- 💾 **实验管理**: 自动管理实验目录和检查点

## 项目结构

```
grpo_qwen_1.5b/
├── README.md
├── requirements.txt
├── config/
│   ├── training.yaml           # 训练配置
│   ├── model.yaml             # 模型配置
│   ├── logging.yaml           # 日志配置
│   └── resume.yaml            # 断点续训配置（自动生成）
├── src/
│   ├── data/
│   │   ├── dataset.py         # 数据集处理
│   │   └── utils.py           # 数据相关工具函数
│   ├── rewards/
│   │   └── rewards.py         # 奖励函数（正确性+格式）
│   ├── grpo/
│   │   └── algorithm.py       # GRPO算法核心实现
│   ├── training/
│   │   └── trainer.py         # 增强训练器（支持断点续训）
│   └── utils/
│       ├── config.py          # 配置管理
│       ├── logging.py         # 日志工具
│       ├── device.py          # 设备管理（支持MPS）
│       └── random.py          # 随机种子设置
├── scripts/
│   ├── train.py               # 训练脚本（支持--resume）
│   ├── evaluate.py            # 评估脚本
│   ├── test_model.py          # 测试微调后的模型
│   ├── visualize_experiment.py # 实验可视化脚本
│   └── download.sh            # 模型和数据下载脚本
└── output/
    ├── experiments/           # 实验目录
    │   └── exp_YYYYMMDD_HHMMSS/
    │       ├── checkpoints/   # 模型检查点
    │       ├── metrics/       # 训练指标
    │       ├── samples/       # 样本输出
    │       └── final_model/   # 最终模型
    ├── logs/                  # 日志文件
    └── plots/                 # 可视化图表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载模型和数据（首次使用）
```bash
# 使用shell脚本自动下载
./scripts/download.sh

# 数据集会自动缓存到 ./datasets/ 目录
# 模型会下载到默认缓存位置
```

### 3. 开始训练
```bash
# 从头开始训练
python scripts/train.py

# 从断点续训
python scripts/train.py --resume
```

### 4. 可视化训练结果
```bash
# 查看训练图表（训练完成后会自动提示命令）
python scripts/visualize_experiment.py output/experiments/exp_20241221_143022
```

### 5. 评估和测试
```bash
# 评估原始模型
python scripts/evaluate.py

# 测试微调后的模型
python scripts/test_model.py
```

## 📊 训练指标说明

### 奖励系统
- **Correctness Reward**: 0-2.0分
  - 2.0分：答案完全正确
  - 1.5分：数值正确但格式不同
  - 0.0分：答案错误
  
- **Format Reward**: 0-0.8分
  - 每个XML标签存在得0.2分：`<reasoning>`, `</reasoning>`, `<answer>`, `</answer>`
  
- **Combined Reward**: 0-2.8分（两者相加）

### 输出指标
- **Loss**: GRPO损失值
- **Average Reward**: 平均组合奖励
- **Output Length**: 生成文本长度统计

## 配置说明

### 训练配置 (config/training.yaml)
- `num_iterations`: 外层迭代次数
- `num_steps`: 每个迭代的训练步数
- `batch_size`: 批次大小
- `num_generations`: 每个提示生成的完成数
- `max_completion_length`: 最大完成长度
- `beta`: KL惩罚系数
- `learning_rate`: 学习率
- `mu`: 每个批次的策略更新次数
- `epsilon`: PPO裁剪参数

### 模型配置 (config/model.yaml)
- `model_name`: 预训练模型名称
- `attn_implementation`: 注意力实现方式
- `torch_dtype`: 数据类型
- `device_map`: 设备映射

## 🔄 断点续训功能

### 自动保存
- 每50步自动保存训练状态到 `config/resume.yaml`
- 保存模型检查点到实验目录
- 记录当前训练进度和最佳奖励

### 续训使用
```bash
# 检查是否有断点可以恢复
ls config/resume.yaml

# 从断点继续训练
python scripts/train.py --resume
```

### 状态恢复
- 自动加载最新检查点模型
- 恢复训练进度（iteration, step）
- 继续记录训练指标历史

## 📈 可视化功能

训练完成后，系统会自动生成以下图表：

- **loss.png**: 训练损失曲线
- **reward.png**: 奖励分解图（正确性+格式+组合）
- **output_length.png**: 输出长度统计
- **loss_vs_reward.png**: 损失vs奖励散点图
- **training_progress.png**: 训练进度综合图
- **summary.json**: 训练统计摘要

```bash
# 手动生成图表
python scripts/visualize_experiment.py output/experiments/exp_20241221_143022 --output_dir output/plots
```

## ⚙️ 硬件要求与配置

### 推荐配置
- **GPU**: 80GB VRAM（如A100）
- **内存**: 32GB+ RAM
- **存储**: 50GB+ 可用空间

### 显存优化
根据你的硬件调整 `config/training.yaml`：

```yaml
# 高显存配置 (80GB)
batch_size: 8
num_generations: 16
max_completion_length: 512

# 中等显存配置 (40GB)  
batch_size: 4
num_generations: 8
max_completion_length: 400

# 低显存配置 (24GB)
batch_size: 2
num_generations: 4
max_completion_length: 256
```

### 设备支持
- ✅ **NVIDIA GPU**: CUDA加速
- ✅ **Apple Silicon**: MPS加速（Mac M1/M2/M3）
- ✅ **CPU**: 纯CPU训练（较慢）

## 🐛 常见问题

### Q: 训练中断了怎么办？
A: 使用 `python scripts/train.py --resume` 从断点继续

### Q: 显存不够怎么办？
A: 减小 `batch_size`、`num_generations` 或 `max_completion_length`

### Q: 如何查看训练进度？
A: 训练过程会实时显示指标，完成后用可视化脚本查看图表

### Q: 模型保存在哪里？
A: 最终模型保存在 `output/experiments/exp_xxx/final_model/`

## 📝 实验记录

每次训练都会创建独立的实验目录：
```
output/experiments/exp_20241221_143022/
├── experiment_config.json    # 实验配置
├── checkpoints/             # 检查点（每50步）
├── metrics/                 # 训练指标JSON
├── samples/                 # 样本输出（每10步）
└── final_model/            # 最终模型
```

