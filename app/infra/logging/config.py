"""
日志配置管理器

根据配置文件设置全局日志系统，支持控制台和文件输出。
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any, Optional
from app.config import get_config_manager


def setup_logging(config_manager=None) -> None:
    """
    设置全局日志配置
    
    Args:
        config_manager: 配置管理器实例，如果为 None 则使用默认配置
    """
    config_manager = config_manager or get_config_manager()
    logging_config = config_manager.get_config().get("logging", {})
    
    # 获取日志级别
    level = logging_config.get("level", "INFO").upper()
    log_level = getattr(logging, level, logging.INFO)
    
    # 获取日志格式
    log_format = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(log_format)
    
    # 配置处理器
    handlers_config = logging_config.get("handlers", {})
    
    # 控制台处理器
    console_config = handlers_config.get("console", {})
    if console_config.get("enabled", True):
        console_handler = logging.StreamHandler()
        console_level = console_config.get("level", "INFO").upper()
        console_handler.setLevel(getattr(logging, console_level, logging.INFO))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 文件处理器
    file_config = handlers_config.get("file", {})
    if file_config.get("enabled", False):
        # 支持按日期创建日志文件
        use_daily_rotation = file_config.get("use_daily_rotation", True)
        
        if use_daily_rotation:
            # 按日期创建日志文件：logs/chocolate_2025-01-15.log
            from datetime import datetime
            date_str = datetime.now().strftime("%Y-%m-%d")
            base_filename = file_config.get("filename", "logs/chocolate.log")
            # 如果文件名包含.log，则替换为带日期的版本
            if base_filename.endswith(".log"):
                filename = base_filename.replace(".log", f"_{date_str}.log")
            else:
                filename = f"{base_filename}_{date_str}.log"
        else:
            filename = file_config.get("filename", "logs/chocolate.log")
        
        # 确保日志目录存在
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据配置选择处理器类型
        if use_daily_rotation:
            # 使用按日期轮转的处理器
            when = file_config.get("rotation_when", "midnight")  # midnight, H, D, W0-W6
            interval = file_config.get("rotation_interval", 1)
            backup_count = file_config.get("backup_count", 30)  # 保留30天的日志
            encoding = file_config.get("encoding", "utf-8")
            
            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=filename,
                when=when,
                interval=interval,
                backupCount=backup_count,
                encoding=encoding
            )
        else:
            # 使用按大小轮转的处理器
            max_bytes = file_config.get("max_bytes", 10 * 1024 * 1024)  # 默认10MB
            backup_count = file_config.get("backup_count", 5)
            encoding = file_config.get("encoding", "utf-8")
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=filename,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding=encoding
            )
        
        file_level = file_config.get("level", "DEBUG").upper()
        file_handler.setLevel(getattr(logging, file_level, logging.DEBUG))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    _configure_third_party_loggers()


def _configure_third_party_loggers() -> None:
    """
    配置第三方库的日志级别，避免过多噪音
    """
    # 设置一些第三方库的日志级别
    third_party_loggers = {
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "httpx": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "PIL": logging.WARNING,
        "matplotlib": logging.WARNING,
        "numpy": logging.WARNING,
        "watchfiles": logging.WARNING,  # 过滤 watchfiles 的 INFO 日志（文件监控相关）
    }
    
    for logger_name, level in third_party_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        
    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    return logging.getLogger(name)


def get_logging_config() -> Dict[str, Any]:
    """
    获取当前日志配置
    
    Returns:
        Dict[str, Any]: 日志配置字典
    """
    config_manager = get_config_manager()
    return config_manager.get_config().get("logging", {})
