"""设备管理工具"""

import torch


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_ids():
    """获取GPU设备ID列表"""
    num_gpus = torch.cuda.device_count()
    return list(range(num_gpus)) if num_gpus > 1 else None


def print_device_info():
    """打印设备信息"""
    device = get_device()
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using primary device: {device}")
        print(f"Detected {num_gpus} GPUs")
        return device, num_gpus
    elif torch.backends.mps.is_available():
        print(f"Using primary device: {device}")
        print("Detected Apple Silicon GPU (MPS)")
        return device, 1
    else:
        print(f"Using primary device: {device}")
        print("No GPU detected")
        return device, 0
