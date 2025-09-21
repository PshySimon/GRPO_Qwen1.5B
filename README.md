# GRPO Qwen2.5-1.5B 项目

基于 GRPO (Group Relative Policy Optimization) 算法微调 Qwen2.5-1.5B-Instruct 模型进行数学问题求解的项目。

## 项目结构

```
grpo_project/
├── README.md
├── requirements.txt
├── config/
│   ├── training.yaml           # 训练配置
│   ├── model.yaml             # 模型配置
│   └── logging.yaml           # 日志配置
├── src/
│   ├── data/
│   │   ├── dataset.py         # 数据集处理
│   │   └── utils.py           # 数据相关工具函数
│   ├── rewards/
│   │   └── rewards.py         # 奖励函数
│   ├── grpo/
│   │   └── algorithm.py       # GRPO算法核心
│   ├── training/
│   │   └── trainer.py         # 训练器
│   └── utils/
│       ├── config.py          # 配置管理
│       ├── logging.py         # 日志工具
│       ├── device.py          # 设备管理
│       └── random.py          # 随机种子设置
├── scripts/
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   └── test_model.py          # 测试微调后的模型
└── output/
    ├── logs/
    ├── models/
    └── results/
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 0. 下载模型和数据（首次使用）

```bash
# 使用shell脚本下载
./scripts/download.sh

# 或者手动下载：
# 1. 模型：从 ModelScope 下载 Qwen2.5-1.5B-Instruct 到 models/qwen2.5-1.5b/
# 2. 数据：从 Hugging Face 下载 GSM8K 数据集到 data/gsm8k/
```

### 1. 训练模型

```bash
python scripts/train.py
```

### 2. 评估模型

```bash
python scripts/evaluate.py
```

### 3. 测试微调后的模型

```bash
python scripts/test_model.py
```

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

## 主要功能

1. **数据处理**: 加载和格式化GSM8K数据集
2. **奖励函数**: 正确性奖励和格式奖励
3. **GRPO算法**: 实现Group Relative Policy Optimization
4. **训练流程**: 完整的训练和评估流程
5. **模型测试**: 测试微调后模型的数学问题求解能力

## 注意事项

- 确保有足够的GPU内存（建议至少80GB VRAM）
- 可以根据硬件配置调整批次大小和生成长度
- 训练过程中会生成详细的日志文件
