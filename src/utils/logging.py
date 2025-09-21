"""日志配置工具"""

import datetime
import logging
from pathlib import Path


def setup_logger(log_dir: str = "./output/logs", level: str = "INFO") -> logging.Logger:
    """设置日志器"""
    # 确保日志目录存在
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 创建logger
    logger = logging.getLogger("GRPOLogger")
    logger.setLevel(getattr(logging, level.upper()))

    # 清除已有的处理器
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 创建文件处理器
    file_handler = logging.FileHandler(
        f'{log_dir}/output.{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log.txt',
        mode="w",
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger initialized")
    return logger
