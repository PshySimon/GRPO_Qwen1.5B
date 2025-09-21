"""配置管理工具"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_name: str, config_dir: str = "config") -> Dict[str, Any]:
    """加载指定的配置文件"""
    config_path = Path(config_dir) / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_all_configs(config_dir: str = "config") -> Dict[str, Any]:
    """加载所有配置文件"""
    configs = {}
    config_path = Path(config_dir)

    for yaml_file in config_path.glob("*.yaml"):
        config_name = yaml_file.stem
        configs[config_name] = load_config(config_name, config_dir)

    return configs
